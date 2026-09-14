#!/usr/bin/env python3
"""
Ultra-Fast Vectorized Snake AI DQN Trainer (Parallel Environments)
Matches train.html's multi-worker throughput in Python.
Saves to brain.json.
"""

import sys
import time
import math
import random
import json
from collections import deque
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

D4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
IN_DIM = 12
OUT_DIM = 3


def rel_dir(dir_idx, a):
    if a == 0:
        return dir_idx
    elif a == 1:
        return (dir_idx + 1) % 4
    else:
        return (dir_idx + 3) % 4


# Fast Flat BFS Flood Fill
if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _reach_kernel(b, occ, start, tc, mark, queue, stamp):
        stamp += 1
        mark[start] = stamp
        queue[0] = start
        head = 0
        tail = 1
        cnt = 0
        tail_seen = False
        while head < tail:
            c = queue[head]
            head += 1
            cnt += 1
            if tc >= 0 and c == tc:
                tail_seen = True
            cx = c % b
            cy = c // b
            if cy > 0:
                u = c - b
                if mark[u] != stamp and (occ[u] == 0 or u == tc):
                    mark[u] = stamp; queue[tail] = u; tail += 1
            if cx < b - 1:
                r = c + 1
                if mark[r] != stamp and (occ[r] == 0 or r == tc):
                    mark[r] = stamp; queue[tail] = r; tail += 1
            if cy < b - 1:
                d = c + b
                if mark[d] != stamp and (occ[d] == 0 or d == tc):
                    mark[d] = stamp; queue[tail] = d; tail += 1
            if cx > 0:
                l = c - 1
                if mark[l] != stamp and (occ[l] == 0 or l == tc):
                    mark[l] = stamp; queue[tail] = l; tail += 1
        return cnt, tail_seen, stamp
else:
    def _reach_kernel(b, occ, start, tc, mark, queue, stamp):
        stamp += 1
        mark[start] = stamp
        queue[0] = start
        head = 0
        tail = 1
        cnt = 0
        tail_seen = False
        while head < tail:
            c = queue[head]
            head += 1
            cnt += 1
            if tc >= 0 and c == tc:
                tail_seen = True
            cx = c % b
            cy = c // b
            if cy > 0:
                u = c - b
                if mark[u] != stamp and (occ[u] == 0 or u == tc):
                    mark[u] = stamp; queue[tail] = u; tail += 1
            if cx < b - 1:
                r = c + 1
                if mark[r] != stamp and (occ[r] == 0 or r == tc):
                    mark[r] = stamp; queue[tail] = r; tail += 1
            if cy < b - 1:
                d = c + b
                if mark[d] != stamp and (occ[d] == 0 or d == tc):
                    mark[d] = stamp; queue[tail] = d; tail += 1
            if cx > 0:
                l = c - 1
                if mark[l] != stamp and (occ[l] == 0 or l == tc):
                    mark[l] = stamp; queue[tail] = l; tail += 1
        return cnt, tail_seen, stamp


class FastSnakeEnv:
    def __init__(self, board=12, max_steps=1200, starve=80, step_cost=0.002, death_penalty=1.0, shape_weight=0.02, apple_reward=1.0):
        self.b = board
        self.n = board * board
        self.cap = self.n + 4
        self.max_steps = max_steps
        self.starve = starve
        self.limit = max(40, starve + self.n // 2)
        self.step_cost = step_cost
        self.death_penalty = death_penalty
        self.shape_weight = shape_weight
        self.apple_reward = apple_reward

        self.occ = np.zeros(self.n, dtype=np.int8)
        self.sx = np.zeros(self.cap, dtype=np.int16)
        self.sy = np.zeros(self.cap, dtype=np.int16)
        self.mark = np.zeros(self.n, dtype=np.int32)
        self.queue = np.zeros(self.n, dtype=np.int32)
        self.stamp = 0
        self.reset()

    def reset(self):
        self.occ.fill(0)
        m = self.b // 2
        self.h = 0
        self.len = 3
        self.dir_idx = 1
        self.steps = 0
        self.hunger = 0
        self.apples = 0
        self.episode_reward = 0.0

        self.sx[0] = m + 1; self.sy[0] = m
        self.sx[1] = m;     self.sy[1] = m
        self.sx[2] = m - 1; self.sy[2] = m
        for i in range(3):
            self.occ[self.sy[i] * self.b + self.sx[i]] = 1

        self.spawn_apple()
        hx, hy = self.sx[self.h], self.sy[self.h]
        self.prev_dist = abs(self.fx - hx) + abs(self.fy - hy)
        return self.sense()

    def tail_cell(self):
        ti = (self.h + self.len - 1) % self.cap
        return self.sy[ti] * self.b + self.sx[ti]

    def spawn_apple(self):
        free_cells = [i for i in range(self.n) if self.occ[i] == 0]
        if not free_cells:
            self.fx, self.fy = -1, -1
            return False
        c = random.choice(free_cells)
        self.fx = c % self.b
        self.fy = c // self.b
        return True

    def reach(self, nx, ny, eats):
        tc = -1 if eats else self.tail_cell()
        start = ny * self.b + nx
        cnt, tail_seen, self.stamp = _reach_kernel(self.b, self.occ, start, tc, self.mark, self.queue, self.stamp)
        return cnt, tail_seen

    def sense(self):
        b = self.b
        hx, hy = self.sx[self.h], self.sy[self.h]
        dir_idx = self.dir_idx
        free_cells = max(1, self.n - self.len)
        tc = self.tail_cell()
        any_tail = 0.0
        inp = np.zeros(12, dtype=np.float32)

        for a in range(3):
            dd = rel_dir(dir_idx, a)
            nx = hx + D4[dd][0]
            ny = hy + D4[dd][1]
            dead = (nx < 0 or ny < 0 or nx >= b or ny >= b)
            eats = False
            if not dead:
                ci = ny * b + nx
                eats = (nx == self.fx and ny == self.fy)
                if self.occ[ci] != 0 and not (not eats and ci == tc):
                    dead = True

            inp[a] = 1.0 if dead else 0.0
            if dead:
                inp[3 + a] = 0.0
            else:
                c, tail_seen = self.reach(nx, ny, eats)
                frac = c / free_cells
                inp[3 + a] = 1.0 if frac > 1.0 else frac
                if tail_seen:
                    any_tail = 1.0

        dx = self.fx - hx
        dy = self.fy - hy
        f = D4[dir_idx]
        r = D4[(dir_idx + 1) % 4]
        inp[6] = (dx * f[0] + dy * f[1]) / b
        inp[7] = (dx * r[0] + dy * r[1]) / b
        inp[8] = (abs(dx) + abs(dy)) / (2 * b)
        inp[9] = self.len / self.n
        hg = self.hunger / self.limit
        inp[10] = 1.0 if hg > 1.0 else hg
        inp[11] = any_tail
        return inp

    def step(self, rel_action):
        action = rel_dir(self.dir_idx, rel_action)
        b = self.b
        cap = self.cap
        ti = (self.h + self.len - 1) % cap
        tail_x, tail_y = self.sx[ti], self.sy[ti]

        hx, hy = self.sx[self.h], self.sy[self.h]
        nx = hx + D4[action][0]
        ny = hy + D4[action][1]
        reward = -self.step_cost

        if nx < 0 or ny < 0 or nx >= b or ny >= b:
            self.dir_idx = action
            reward -= self.death_penalty
            self.episode_reward += reward
            return reward, True, True, 'wall', False

        eats = (nx == self.fx and ny == self.fy)
        nci = ny * b + nx
        if self.occ[nci] != 0 and not (not eats and nx == tail_x and ny == tail_y):
            self.dir_idx = action
            reward -= self.death_penalty
            self.episode_reward += reward
            return reward, True, True, 'body', False

        new_dist = abs(self.fx - nx) + abs(self.fy - ny)
        reward += (self.prev_dist - new_dist) * self.shape_weight
        self.prev_dist = new_dist

        if not eats:
            self.occ[tail_y * b + tail_x] = 0
        else:
            self.len += 1

        self.h = (self.h + cap - 1) % cap
        self.sx[self.h] = nx
        self.sy[self.h] = ny
        self.occ[nci] = 1
        self.dir_idx = action
        self.steps += 1
        self.hunger += 1

        done = False
        death = None
        terminal = False

        if eats:
            self.apples += 1
            self.hunger = 0
            reward += self.apple_reward
            if not self.spawn_apple():
                done = True
                terminal = True
                death = 'win'
            else:
                self.prev_dist = abs(self.fx - nx) + abs(self.fy - ny)

        if not done and self.hunger > self.limit:
            done = True
            death = 'starve'

        if not done and self.steps >= self.max_steps:
            done = True
            death = 'cap'

        self.episode_reward += reward
        return reward, done, terminal, death, eats


# --- Prioritized Class Replay Buffer ---

class PrioritizedClassReplayBuffer:
    def __init__(self, cap=60000, alpha=0.5, beta=0.5):
        self.cap = cap
        self.alpha = alpha
        self.beta = beta
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((cap, IN_DIM), dtype=np.float32)
        self.ns = np.zeros((cap, IN_DIM), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.g = np.zeros(cap, dtype=np.float32)
        self.cls = np.zeros(cap, dtype=np.int32)
        self.p = np.zeros(cap, dtype=np.float32)
        self.max_p = 1.0

        self.class_quotas = [0.30, 0.28, 0.14, 0.28]
        self.class_map = [1, 2, 3, 0]

    def push(self, s, a, r, ns, d, g, cls):
        idx = self.ptr
        self.s[idx] = s
        self.ns[idx] = ns
        self.a[idx] = a
        self.r[idx] = r
        self.d[idx] = d
        self.g[idx] = g
        self.cls[idx] = cls
        self.p[idx] = max(self.max_p, 1e-3)

        self.ptr = (self.ptr + 1) % self.cap
        if self.size < self.cap:
            self.size += 1

    def sample(self, batch_size):
        indices = np.empty(batch_size, dtype=np.int64)
        size = self.size

        for i in range(batch_size):
            if random.random() < 0.30:
                indices[i] = random.randint(0, size - 1)
            else:
                q = random.random()
                acc = 0.0
                target_cls = 0
                for c in range(4):
                    acc += self.class_quotas[c]
                    if q <= acc:
                        target_cls = self.class_map[c]
                        break

                best_idx = -1
                best_p = -1.0
                for _ in range(8):
                    rand_idx = random.randint(0, size - 1)
                    if self.cls[rand_idx] == target_cls and self.p[rand_idx] > best_p:
                        best_idx = rand_idx
                        best_p = self.p[rand_idx]
                if best_idx < 0:
                    best_idx = random.randint(0, size - 1)
                indices[i] = best_idx

        mean_p = self.max_p * 0.35 + 1e-3
        weights = np.clip((mean_p / np.maximum(self.p[indices], 1e-4)) ** self.beta, 0.25, 2.5).astype(np.float32)

        return (
            self.s[indices],
            self.a[indices],
            self.r[indices],
            self.ns[indices],
            self.d[indices],
            self.g[indices],
            weights,
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            p = (abs(td) + 1e-3) ** self.alpha
            self.p[idx] = p
            if p > self.max_p:
                self.max_p = p * 0.999 + self.max_p * 0.001


# --- Neural Network Model ---

if TORCH_AVAILABLE:
    class TorchDQN(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.hidden = hidden
            self.fc1 = nn.Linear(IN_DIM, hidden)
            self.fc2 = nn.Linear(hidden, OUT_DIM)
            nn.init.normal_(self.fc1.weight, std=0.15)
            nn.init.zeros_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight, std=0.15)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x):
            return self.fc2(torch.tanh(self.fc1(x)))

        def get_flat_weights(self):
            w1 = self.fc1.weight.detach().cpu().numpy().flatten()
            b1 = self.fc1.bias.detach().cpu().numpy().flatten()
            w2 = self.fc2.weight.detach().cpu().numpy().flatten()
            b2 = self.fc2.bias.detach().cpu().numpy().flatten()
            return np.concatenate([w1, b1, w2, b2]).tolist()
else:
    class NumpyDQN:
        def __init__(self, hidden=32, lr=1e-4):
            self.hidden = hidden
            self.lr = lr
            self.w1 = (np.random.randn(hidden, IN_DIM) * 0.15).astype(np.float32)
            self.b1 = np.zeros(hidden, dtype=np.float32)
            self.w2 = (np.random.randn(OUT_DIM, hidden) * 0.15).astype(np.float32)
            self.b2 = np.zeros(OUT_DIM, dtype=np.float32)

            self.m_w1 = np.zeros_like(self.w1); self.v_w1 = np.zeros_like(self.w1)
            self.m_b1 = np.zeros_like(self.b1); self.v_b1 = np.zeros_like(self.b1)
            self.m_w2 = np.zeros_like(self.w2); self.v_w2 = np.zeros_like(self.w2)
            self.m_b2 = np.zeros_like(self.b2); self.v_b2 = np.zeros_like(self.b2)
            self.t = 0

        def forward(self, x):
            h = np.tanh(x @ self.w1.T + self.b1)
            return h @ self.w2.T + self.b2, h

        def copy_from(self, other):
            self.w1[:] = other.w1; self.b1[:] = other.b1
            self.w2[:] = other.w2; self.b2[:] = other.b2

        def get_flat_weights(self):
            return np.concatenate([self.w1.flatten(), self.b1, self.w2.flatten(), self.b2]).tolist()

        def train_step(self, x, a, target, weights):
            self.t += 1
            B = x.shape[0]
            q, h = self.forward(x)
            q_a = q[np.arange(B), a]
            err = q_a - target
            clipped_err = np.clip(err, -1.0, 1.0) * weights

            grad_q = np.zeros_like(q)
            grad_q[np.arange(B), a] = clipped_err / B

            gw2 = grad_q.T @ h
            gb2 = np.sum(grad_q, axis=0)
            gh = grad_q @ self.w2
            gz = gh * (1.0 - h * h)
            gw1 = gz.T @ x
            gb1 = np.sum(gz, axis=0)

            b1, b2, eps = 0.9, 0.999, 1e-8
            for w, dw, m, v in [(self.w1, gw1, self.m_w1, self.v_w1),
                               (self.b1, gb1, self.m_b1, self.v_b1),
                               (self.w2, gw2, self.m_w2, self.v_w2),
                               (self.b2, gb2, self.m_b2, self.v_b2)]:
                m[:] = b1 * m + (1 - b1) * dw
                v[:] = b2 * v + (1 - b2) * (dw ** 2)
                mh = m / (1.0 - b1 ** self.t)
                vh = v / (1.0 - b2 ** self.t)
                w -= self.lr * mh / (np.sqrt(vh) + eps)

            return err


def select_greedy_action(state_obs, q_values):
    best_a = -1
    for a in range(3):
        if state_obs[a] >= 0.5:
            continue
        if best_a < 0:
            best_a = a
            continue
        diff = q_values[a] - q_values[best_a]
        if diff > 1e-6 or (abs(diff) <= 1e-6 and state_obs[3 + a] > state_obs[3 + best_a]):
            best_a = a
    if best_a < 0:
        best_a = int(np.argmax(q_values))
    return best_a


def select_explore_action(state_obs):
    if random.random() < 0.15:
        return random.randint(0, 2)
    weights = [0.02 if state_obs[a] >= 0.5 else (0.15 + state_obs[3 + a]) for a in range(3)]
    tot = sum(weights)
    rnd = random.random() * tot
    for a in range(3):
        rnd -= weights[a]
        if rnd <= 0:
            return a
    return 2


def save_brain_json(model, hidden, episode, best_apples, board=12, filepath="brain.json"):
    data = {
        "format": "snake-ai-lab-brain",
        "version": 1,
        "hidden": hidden,
        "inputs": IN_DIM,
        "outputs": OUT_DIM,
        "gen": episode,
        "apples": best_apples,
        "fitness": best_apples,
        "board": board,
        "saved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights": [round(float(w), 5) for w in model.get_flat_weights()]
    }
    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"\n>>> [SAVED CHECKPOINT] {filepath} (Episode {episode}, Record: {best_apples} apples)")


class Visualizer:
    def __init__(self, board_size=12, cell_size=28):
        self.b = board_size
        self.cell = cell_size
        self.sidebar = 250
        self.w = self.b * self.cell + self.sidebar
        self.h = self.b * self.cell
        pygame.init()
        pygame.display.set_caption("Snake AI — Turbo Trainer (SPACE = Turbo/Live)")
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.font = pygame.font.SysFont("consolas", 14)
        self.clock = pygame.time.Clock()
        self.turbo = True
        self.last_draw = time.time()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.turbo = not self.turbo
                    print(f"\n[Visualizer] Switched -> {'TURBO (Max Speed)' if self.turbo else 'LIVE PLAY (60 FPS)'}")

    def render(self, env, q_values, ep, apples, best, eps, sps):
        now = time.time()
        if self.turbo and (now - self.last_draw < 0.066):
            return

        self.handle_events()
        self.last_draw = now

        self.screen.fill((11, 15, 20))

        for y in range(self.b):
            for x in range(self.b):
                r = pygame.Rect(x * self.cell, y * self.cell, self.cell - 1, self.cell - 1)
                pygame.draw.rect(self.screen, (18, 25, 34), r)

        if env.fx >= 0:
            ar = pygame.Rect(env.fx * self.cell + 3, env.fy * self.cell + 3, self.cell - 6, self.cell - 6)
            pygame.draw.rect(self.screen, (251, 113, 133), ar, border_radius=5)

        cap = env.cap
        for i in range(env.len):
            idx = (env.h + i) % cap
            bx, by = env.sx[idx], env.sy[idx]
            sr = pygame.Rect(bx * self.cell + 2, by * self.cell + 2, self.cell - 4, self.cell - 4)
            color = (45, 212, 191) if i == 0 else (20, 150, 135)
            pygame.draw.rect(self.screen, color, sr, border_radius=4)

        sx = self.b * self.cell + 16
        lines = [
            f"EPISODE:  {ep}",
            f"APPLES:   {apples} (Best: {best})",
            f"SPEED:    {sps:.0f} steps/sec",
            f"EPSILON:  {eps:.3f}",
            "",
            "Q-VALUES:",
            f" Straight: {q_values[0]:.2f}",
            f" Right:    {q_values[1]:.2f}",
            f" Left:     {q_values[2]:.2f}",
            "",
            "MODE [SPACE]:",
            " >>> TURBO SPEED <<<" if self.turbo else " LIVE 60 FPS",
        ]
        y_off = 20
        for line in lines:
            t = self.font.render(line, True, (234, 241, 248))
            self.screen.blit(t, (sx, y_off))
            y_off += 22

        pygame.display.flip()
        if not self.turbo:
            self.clock.tick(60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="High-Performance Parallel Snake DQN Trainer")
    parser.add_argument("--board", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel snake environments")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--nstep", type=int, default=3)
    parser.add_argument("--train-freq", type=int, default=4, help="Updates every N vectorized steps")
    parser.add_argument("--gui", action="store_true", help="Launch live Pygame visualizer")
    args = parser.parse_args()

    num_envs = args.workers
    engine = "PyTorch" if TORCH_AVAILABLE else "NumPy"
    numba_status = "Enabled" if NUMBA_AVAILABLE else "Not installed (pip install numba for extra speed)"
    device_str = "CUDA (GPU)" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "CPU"

    print("=" * 65)
    print(f"Snake DQN Trainer | Engine: {engine} ({device_str})")
    print(f"Parallel Environments: {num_envs} snakes | Train Freq: 1 per {args.train_freq} env ticks")
    print(f"Board: {args.board}x{args.board} | Hidden Units: {args.hidden} | Batch: {args.batch}")
    print(f"Target Output: brain.json")
    print("=" * 65)

    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        online_net = TorchDQN(args.hidden).to(device)
        target_net = TorchDQN(args.hidden).to(device)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    else:
        online_net = NumpyDQN(args.hidden, lr=args.lr)
        target_net = NumpyDQN(args.hidden, lr=args.lr)
        target_net.copy_from(online_net)

    replay = PrioritizedClassReplayBuffer(cap=60000)

    # Initialize parallel environments
    envs = [FastSnakeEnv(board=args.board) for _ in range(num_envs)]
    obs_batch = np.array([env.reset() for env in envs], dtype=np.float32)

    # N-step transition window per environment
    worker_q = [{"s": [], "a": [], "r": [], "apple": [], "danger": []} for _ in range(num_envs)]

    viz = None
    if args.gui:
        if PYGAME_AVAILABLE:
            viz = Visualizer(board_size=args.board)
            print("[Visualizer] Running in TURBO. Press SPACE inside window for live playback.")
        else:
            print("[Warning] Pygame not installed. Running in headless terminal mode.")

    epsilon = 1.0
    eps_end = 0.02
    eps_decay_steps = 250000
    target_sync_updates = 1500

    episodes = 0
    total_steps = 0
    updates = 0
    best_apples = 0
    recent_apples = deque(maxlen=100)

    def flush_worker_window(w_idx, ns_obs, terminal, full=False):
        wq = worker_q[w_idx]
        count = 1 if full else len(wq["s"])
        for _ in range(count):
            k = len(wq["s"])
            R = 0.0
            disc = 1.0
            had_apple = False
            for i in range(k):
                R += disc * wq["r"][i]
                disc *= args.gamma
                if wq["apple"][i]:
                    had_apple = True

            cls = 2 if terminal else (1 if had_apple else (3 if wq["danger"][0] else 0))
            replay.push(wq["s"].pop(0), wq["a"].pop(0), R, ns_obs, 1.0 if terminal else 0.0, disc, cls)
            if full:
                break

    t0 = time.time()
    sps = 0.0

    try:
        while True:
            # Batch inference for all parallel snakes simultaneously
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    t_obs = torch.from_numpy(obs_batch).to(device)
                    q_vals_batch = online_net(t_obs).cpu().numpy()
            else:
                q_vals_batch, _ = online_net.forward(obs_batch)

            # Step each parallel environment
            for i in range(num_envs):
                total_steps += 1
                if total_steps < eps_decay_steps:
                    epsilon = 1.0 - (1.0 - eps_end) * (total_steps / eps_decay_steps)
                else:
                    epsilon = eps_end

                obs = obs_batch[i]
                q_vals = q_vals_batch[i]

                if random.random() < epsilon:
                    action = select_explore_action(obs)
                else:
                    action = select_greedy_action(obs, q_vals)

                s_copy = np.copy(obs)
                danger = 1 if (obs[0] > 0.5 or obs[1] > 0.5 or obs[2] > 0.5) else 0

                reward, done, terminal, death, eats = envs[i].step(action)

                worker_q[i]["s"].append(s_copy)
                worker_q[i]["a"].append(action)
                worker_q[i]["r"].append(reward)
                worker_q[i]["apple"].append(eats)
                worker_q[i]["danger"].append(danger)

                next_obs = envs[i].sense()

                if done:
                    flush_worker_window(i, next_obs, terminal, full=False)
                    worker_q[i]["s"].clear(); worker_q[i]["a"].clear(); worker_q[i]["r"].clear()
                    worker_q[i]["apple"].clear(); worker_q[i]["danger"].clear()

                    episodes += 1
                    recent_apples.append(envs[i].apples)
                    if envs[i].apples > best_apples:
                        best_apples = envs[i].apples
                        save_brain_json(online_net, args.hidden, episodes, best_apples, board=args.board, filepath="brain.json")

                    obs_batch[i] = envs[i].reset()
                else:
                    if len(worker_q[i]["s"]) >= args.nstep:
                        flush_worker_window(i, next_obs, False, full=True)
                    obs_batch[i] = next_obs

            # Train network once per vectorized batch tick
            if replay.size >= args.batch and (total_steps // num_envs) % args.train_freq == 0:
                b_s, b_a, b_r, b_ns, b_d, b_g, b_w, b_indices = replay.sample(args.batch)

                if TORCH_AVAILABLE:
                    ts_s = torch.from_numpy(b_s).to(device)
                    ts_a = torch.from_numpy(b_a).unsqueeze(1).to(device)
                    ts_r = torch.from_numpy(b_r).to(device)
                    ts_ns = torch.from_numpy(b_ns).to(device)
                    ts_d = torch.from_numpy(b_d).to(device)
                    ts_g = torch.from_numpy(b_g).to(device)
                    ts_w = torch.from_numpy(b_w).to(device)

                    curr_q = online_net(ts_s).gather(1, ts_a).squeeze(1)

                    with torch.no_grad():
                        next_q_online = online_net(ts_ns)
                        dead_mask = (ts_ns[:, :3] >= 0.5)
                        masked_q = next_q_online.clone()
                        masked_q[dead_mask] = -1e9
                        best_actions = torch.argmax(masked_q, dim=1, keepdim=True)

                        next_q_target = target_net(ts_ns).gather(1, best_actions).squeeze(1)
                        target = ts_r + (1.0 - ts_d) * ts_g * next_q_target

                    td = curr_q - target
                    loss = (0.5 * torch.clamp(td.abs(), max=1.0) ** 2 * ts_w).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=5.0)
                    optimizer.step()

                    replay.update_priorities(b_indices, td.detach().cpu().numpy())
                    updates += 1

                    if updates % target_sync_updates == 0:
                        target_net.load_state_dict(online_net.state_dict())
                else:
                    next_q_online, _ = online_net.forward(b_ns)
                    dead_mask = (b_ns[:, :3] >= 0.5)
                    next_q_online[dead_mask] = -1e9
                    best_actions = np.argmax(next_q_online, axis=1)

                    next_q_target, _ = target_net.forward(b_ns)
                    target = b_r + (1.0 - b_d) * b_g * next_q_target[np.arange(args.batch), best_actions]

                    td = online_net.train_step(b_s, b_a, target, b_w)
                    replay.update_priorities(b_indices, td)
                    updates += 1

                    if updates % target_sync_updates == 0:
                        target_net.copy_from(online_net)

            if viz:
                # Render Environment 0 in the GUI
                viz.render(envs[0], q_vals_batch[0], episodes, envs[0].apples, best_apples, epsilon, sps)

            if episodes > 0 and episodes % 50 == 0:
                elapsed = time.time() - t0
                sps = total_steps / max(1e-3, elapsed)
                avg_app = sum(recent_apples) / max(1, len(recent_apples))
                print(f"Ep {episodes:5d} | Steps: {total_steps:7d} ({sps:.0f} steps/s) | Avg Apples: {avg_app:.1f} | Best: {best_apples:2d} | Eps: {epsilon:.3f}")

            if episodes > 0 and episodes % 500 == 0:
                save_brain_json(online_net, args.hidden, episodes, best_apples, board=args.board, filepath="brain.json")

    except KeyboardInterrupt:
        print("\n[Stopped by User] Saving latest brain checkpoint to brain.json...")
        save_brain_json(online_net, args.hidden, episodes, best_apples, board=args.board, filepath="brain.json")


if __name__ == "__main__":
    main()
