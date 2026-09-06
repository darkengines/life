"""Asynchronous training of the world's shared perception encoder.

Runs as its OWN process on the GPU, entirely off the simulation's tick loop:
it reads the experience chunks the simulation has already written, trains,
writes improved encoder weights to a file, and the simulation hot-loads them
whenever they change. Neither side ever waits for the other, which is the
requirement -- the world keeps running at full speed while perception
improves in the background.

WHY THIS OBJECTIVE
Every creature has its own evolved decision layer, but perception is shared,
so the encoder is the one part that can be trained on the whole population's
experience at once. It is trained self-supervised, to predict what happens
next: given the latent of the current senses and the action actually taken,
predict the next sense vector, and predict the reward. No hand-designed
notion of "good behaviour" is involved -- a representation that supports
predicting consequences is a representation that carries the information
control needs, and it cannot bias evolution toward any particular strategy.

This matters because measurement showed evolved brains performing no better
than random ones at most behaviours: each individual was expected to discover
feature extraction from 34 raw channels by mutation alone. Learning the
representation once, from data, is the part that gradient descent is
genuinely better at than evolution.
"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "var"
LOG_DIR = VAR / "experience_log"
WEIGHTS_PATH = VAR / "shared_encoder.npz"
STATUS_PATH = VAR / "trainer_status.json"

LATENT_DIM = 12
TRAIN_INTERVAL = 60.0     # seconds between training rounds
STEPS_PER_ROUND = 400
BATCH = 4096
LR = 1e-3
MIN_ROWS = 4000           # don't bother training on a trickle
# Train on a RECENT window rather than everything ever logged. The world is
# not stationary -- bodies get larger, senses come and go as organs evolve,
# whole strategies rise and fall -- so old transitions describe a world that
# no longer exists. Measured directly: with an ever-growing buffer the
# encoder's advantage over the naive baseline decayed round by round
# (3.8x -> 3.0x -> 2.2x -> 2.0x) as it was forced to fit stale distributions.
RECENT_CHUNKS = 8
# Reward is far sparser and larger in scale than the sense vector, so it needs
# a modest weight or it dominates the representation the encoder learns.
REWARD_LOSS_WEIGHT = 0.05
# Reproduction is decisive but fires on ~0.5% of rows; energy change is dense
# and available on every one. Both feed the learning signal.
# Reproduction is fitness; energy is only a means to it. The first weighting
# had energy dense on every row and reproduction firing on ~0.5% of them, so
# energy dominated the objective roughly a hundredfold and the policy learned
# to HOARD: with instinct at 0.60 strength, mean energy tripled to 182 while
# births fell 73% (7992 -> 2152) and bodies shrank. Higher standing population,
# but a worse life history. Reproduction is now weighted to dominate, and
# energy kept only as a weak shaping term toward it.
REPRO_SIGNAL_WEIGHT = 60.0
ENERGY_SIGNAL_SCALE = 0.05
# Discount for crediting a reward back through the ticks that produced it.
RETURN_DISCOUNT = 0.9
# Advantage-weighted regression temperature and how hard the policy pulls on
# the shared representation.
AWR_TEMPERATURE = 1.0
POLICY_LOSS_WEIGHT = 0.5


HIDDEN_DIM = 12   # must match individuals.rs HIDDEN_DIM


class WorldModel(nn.Module):
    """encoder(sense) -> latent, plus prediction heads and a baseline policy.

    Two things are exported to the simulation:

    * the ENCODER, which becomes every creature's shared perception;
    * the POLICY, latent -> action, which newborns inherit as instinct and
      then evolve away from individually.

    The policy deliberately has exactly the shape of an individual's own
    decoder (latent -> hidden -> action, tanh throughout), because it has to
    be installable as one. The prediction heads are internal -- they exist
    only to give the encoder a reason to retain useful information.
    """

    def __init__(self, sense_dim, act_dim, latent_dim):
        super().__init__()
        # The exported encoder is deliberately a single tanh layer, because
        # that is exactly what the engine evaluates per creature per tick.
        # Anything deeper would not survive the trip back.
        self.encoder = nn.Linear(sense_dim, latent_dim)
        self.predict_next = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 128), nn.ReLU(),
            nn.Linear(128, sense_dim),
        )
        self.predict_reward = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.policy_l1 = nn.Linear(latent_dim, HIDDEN_DIM)
        self.policy_l2 = nn.Linear(HIDDEN_DIM, act_dim)

    def forward(self, sense, action):
        z = torch.tanh(self.encoder(sense))
        za = torch.cat([z, action], dim=-1)
        return z, self.predict_next(za), self.predict_reward(za).squeeze(-1)

    def policy(self, z):
        return torch.tanh(self.policy_l2(torch.tanh(self.policy_l1(z))))


def load_transitions(max_files=RECENT_CHUNKS):
    """Consecutive (state, action, reward, next state) tuples.

    The engine logs each individual for a short run of CONSECUTIVE ticks, so
    rows that share an id and sit one tick apart are a genuine transition.
    Anything else is dropped rather than silently pairing unrelated moments.
    """
    files = sorted(glob.glob(str(LOG_DIR / "*.npz")))[-max_files:]
    target_shape = None
    S, A, R, S2 = [], [], [], []
    for f in reversed(files):
        try:
            d = np.load(f)
        except Exception:
            continue
        ids, ticks = d["ids"], d["ticks"]
        sense, action = d["sense"], d["action"]
        shape = (sense.shape[1], action.shape[1])
        if target_shape is None:
            target_shape = shape
        elif shape != target_shape:
            continue
        reward = d["reward"] if "reward" in d.files else np.zeros(len(ids), dtype=np.float32)
        energy = d["energy"] if "energy" in d.files else np.zeros(len(ids), dtype=np.float32)
        order = np.lexsort((ticks, ids))
        ids, ticks = ids[order], ticks[order]
        sense, action, reward = sense[order], action[order], reward[order]
        energy = energy[order]
        same = (ids[1:] == ids[:-1]) & (ticks[1:] == ticks[:-1] + 1)
        idx = np.nonzero(same)[0]
        if len(idx):
            S.append(sense[idx]); A.append(action[idx])
            S2.append(sense[idx + 1])
            # Reproduction is the sparse, decisive reward, but on its own it
            # fires on ~0.5% of rows -- far too rare to shape a policy. The
            # change in energy across the step is a dense signal available on
            # every row: gaining energy means the creature just ate, losing it
            # means it is paying to exist. Combining them gives something
            # learnable that still treats reproduction as what matters most.
            denergy = energy[idx + 1] - energy[idx]
            step_r = (reward[idx] * REPRO_SIGNAL_WEIGHT
                      + denergy * ENERGY_SIGNAL_SCALE).astype(np.float32)
            # Credit reproduction BACKWARD over the logged trajectory, so the
            # actions that led to breeding are reinforced rather than only the
            # single tick it happened on. Without this, a reward that fires on
            # ~0.5% of rows teaches almost nothing about how it was earned.
            # Rows here are consecutive and sorted by (id, tick), so one
            # reverse discounted pass gives each step the return that followed.
            step_ids, step_ticks = ids[idx], ticks[idx]
            for j in range(len(step_r) - 2, -1, -1):
                if step_ids[j] == step_ids[j + 1] and step_ticks[j + 1] == step_ticks[j] + 1:
                    step_r[j] += RETURN_DISCOUNT * step_r[j + 1]
            R.append(step_r)
    if not S:
        return None
    return (np.concatenate(S), np.concatenate(A),
            np.concatenate(R), np.concatenate(S2))


def write_status(**kw):
    kw["updated"] = time.time()
    try:
        STATUS_PATH.write_text(json.dumps(kw, indent=2))
    except OSError:
        pass


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[trainer] device={dev} "
          f"({torch.cuda.get_device_name(0) if dev == 'cuda' else 'no GPU'})", flush=True)
    model, opt, round_no = None, None, 0

    while True:
        data = load_transitions()
        if data is None or len(data[0]) < MIN_ROWS:
            n = 0 if data is None else len(data[0])
            print(f"[trainer] only {n} transitions available, waiting", flush=True)
            write_status(state="waiting", transitions=n)
            time.sleep(TRAIN_INTERVAL)
            continue

        S, A, R, S2 = data
        sense_dim, act_dim = S.shape[1], A.shape[1]
        if model is None:
            model = WorldModel(sense_dim, act_dim, LATENT_DIM).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            print(f"[trainer] model {sense_dim}->{LATENT_DIM}, act {act_dim}", flush=True)

        # Guard against a single pathological row wrecking a round: the
        # simulation is a live system and a malformed or extreme value should
        # degrade training, not destroy it.
        R = np.clip(np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S2 = np.nan_to_num(S2, nan=0.0, posinf=0.0, neginf=0.0)

        S_t = torch.from_numpy(S).float().to(dev)
        A_t = torch.from_numpy(A).float().to(dev)
        R_t = torch.from_numpy(R).float().to(dev)
        S2_t = torch.from_numpy(S2).float().to(dev)

        model.train()
        last_state, last_reward, last_policy = 0.0, 0.0, 0.0
        for step in range(STEPS_PER_ROUND):
            i = torch.randint(0, len(S_t), (min(BATCH, len(S_t)),), device=dev)
            _z, pred_next, pred_r = model(S_t[i], A_t[i])
            # Predict the CHANGE, not the next absolute state. Predicting the
            # absolute vector is dominated by "it is almost the same as now",
            # which a 12-dimensional latent bottleneck physically cannot
            # reproduce -- so the model scored WORSE than a do-nothing
            # baseline (0.67x) while actually learning fine. The delta is the
            # part that carries information, and it forces the latent to
            # encode what varies rather than spending capacity echoing the
            # present.
            state_loss = nn.functional.mse_loss(pred_next, S2_t[i] - S_t[i])
            reward_loss = nn.functional.mse_loss(pred_r, R_t[i])
            # Advantage-weighted regression: imitate the actions that were
            # actually followed by good outcomes, weighted by how good. This
            # is deliberately imitation of the population's OWN behaviour
            # rather than any designed notion of correct play -- it distils
            # what already works in this world, so it cannot push evolution
            # toward a strategy the world does not reward.
            adv = (R_t[i] - R_t[i].mean()) / (R_t[i].std() + 1e-6)
            wgt = torch.exp(torch.clamp(adv / AWR_TEMPERATURE, -3.0, 3.0)).detach()
            pol = model.policy(_z.detach())
            policy_loss = (wgt * ((pol - A_t[i]) ** 2).mean(dim=-1)).mean()
            loss = state_loss + REWARD_LOSS_WEIGHT * reward_loss + POLICY_LOSS_WEIGHT * policy_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_state = float(state_loss.item())
            last_reward = float(reward_loss.item())
            last_policy = float(policy_loss.item())

        # The only honest baseline for the state head is "predict no change".
        # Reporting the COMBINED loss against it was apples-to-oranges: the
        # combined figure carries the reward term too, so a perfectly healthy
        # state prediction looked like a regression.
        with torch.no_grad():
            # Matching baseline: predicting no change at all.
            naive = float((S2_t - S_t).pow(2).mean().item())
            reward_var = float(R_t.var().item())

        def cpu(t):
            return t.detach().cpu().numpy().astype(np.float32)

        tmp = WEIGHTS_PATH.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            sense_dim=np.array([sense_dim], dtype=np.int32),
            act_dim=np.array([act_dim], dtype=np.int32),
            latent_dim=np.array([LATENT_DIM], dtype=np.int32),
            hidden_dim=np.array([HIDDEN_DIM], dtype=np.int32),
            w=cpu(model.encoder.weight).reshape(-1),
            b=cpu(model.encoder.bias),
            # The learned instinct, in exactly an individual decoder's shape.
            pw1=cpu(model.policy_l1.weight).reshape(-1),
            pb1=cpu(model.policy_l1.bias),
            pw2=cpu(model.policy_l2.weight).reshape(-1),
            pb2=cpu(model.policy_l2.bias),
        )
        os.replace(tmp, WEIGHTS_PATH)   # atomic, so the sim never reads a half-written file

        round_no += 1
        skill = naive / last_state if last_state > 1e-12 else float("inf")
        print(f"[trainer] round {round_no}: {len(S_t)} transitions | "
              f"state {last_state:.6f} vs naive {naive:.6f} ({skill:.2f}x) | "
              f"reward {last_reward:.5f} vs var {reward_var:.5f} | "
              f"policy {last_policy:.5f} -> weights written",
              flush=True)
        write_status(state="trained", round=round_no, transitions=int(len(S_t)),
                     state_loss=last_state, naive_loss=naive, skill_vs_naive=skill,
                     reward_loss=last_reward, reward_var=reward_var,
                     policy_loss=last_policy, device=dev)
        if os.environ.get("TRAIN_ONCE"):
            break
        time.sleep(TRAIN_INTERVAL)


if __name__ == "__main__":
    main()
