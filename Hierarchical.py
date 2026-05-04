from __future__ import annotations
import warnings
warnings.filterwarnings('ignore', message='A JAX array is being set as static')
from dataclasses import dataclass
from typing import Any
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from equinox import tree_at
from pymdp.agent import Agent
from pymdp.control import Policies
from pymdp.maths import calc_vfe

key = jr.PRNGKey(0)



def normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-16) -> jnp.ndarray:
    x = jnp.asarray(x)
    x = jnp.clip(x, a_min=eps)
    return x / x.sum(axis=axis, keepdims=True)


def log_stable(x: jnp.ndarray, eps: float = 1e-16) -> jnp.ndarray:
    return jnp.log(jnp.clip(x, a_min=eps))


def entropy(p: jnp.ndarray, eps: float = 1e-16) -> float:
    return float(-jnp.sum(p * log_stable(p, eps=eps)))


def sample_categorical(rng_key: jnp.ndarray, probs: jnp.ndarray) -> int:
    return int(jr.categorical(rng_key, jnp.log(probs))) # 랜덤 샘플링으로 얻은 index


# A1, B1 learning
def stack_beliefs_over_time(qs_list: list[list[jnp.ndarray]]) -> list[jnp.ndarray]:
    num_factors = len(qs_list[0])
    out: list[jnp.ndarray] = []
    for f in range(num_factors):
        out.append(jnp.concatenate([qs_t[f] for qs_t in qs_list], axis=1))
    return out

# Epistemic & Extrinsic value 출력용
def compute_EFE_components(
    agent: Agent,
    qs: list[jnp.ndarray],
    q_pi: jnp.ndarray,
) -> tuple[float, float]:
    """
    q_pi-weighted extrinsic & epistemic value를 계산.

    Extrinsic (pragmatic) value : E_π[ Σ_t Σ_m C_m · q(o_t,m) ]
        → 선호 관측을 얼마나 기대하는가 (C 벡터와의 내적)
    Epistemic (info gain) value  : E_π[ E_s[H[P(o_m|s)]] - Σ_t Σ_m H[q(o_m)] ]
        → 관측이 상태 불확실성을 얼마나 줄이는가 (= mutual information)
    """
    policies  = agent.policies.policy_arr   # (num_policies, T, num_factors)
    q_pi_vec  = q_pi[0]             # (num_policies,)
    B_deps    = agent.B_dependencies
    A_deps    = agent.A_dependencies

    ext_list, epi_list = [], []

    for pi_idx in range(policies.shape[0]):
        ext, epi = 0.0, 0.0
        qs_t   = [q[0, -1] for q in qs]  # 현재 belief
        policy = policies[pi_idx]         # (T, num_factors)

        for t in range(policy.shape[0]):
            actions = policy[t]

            # ---- B로 다음 상태 예측 ----
            qs_pred = []
            for f in range(len(agent.B)):
                a   = int(actions[f])
                B_f = agent.B[f][0]       # batch dim 제거
                deps = B_deps[f]

                if len(deps) == 1:
                    qn = normalize(B_f[:, :, a] @ qs_t[deps[0]])
                else:  # e.g. heart: deps=[1,0], B_f:(ns,ns_h,ns_s,na)
                    qn = normalize(jnp.einsum(
                        'ijk,j,k->i', B_f[:, :, :, a], qs_t[deps[0]], qs_t[deps[1]]))
                qs_pred.append(qn)

            # ---- 모달리티별 extrinsic / epistemic 계산 ----
            for m, deps in enumerate(A_deps):
                A_m = agent.A[m][0]   # batch dim 제거
                C_m = agent.C[m][0]   # (num_obs,) log 선호

                if len(deps) == 1:
                    qo  = A_m @ qs_pred[deps[0]]
                    H_s = -(A_m * log_stable(A_m)).sum(0)          # (ns,)
                    amb = float((qs_pred[deps[0]] * H_s).sum())
                else:  # len == 2
                    qo  = jnp.einsum('ijk,j,k->i', A_m, qs_pred[deps[0]], qs_pred[deps[1]])
                    H_s = -(A_m * log_stable(A_m)).sum(0)          # (ns1, ns2)
                    amb = float(jnp.einsum('jk,j,k->', H_s, qs_pred[deps[0]], qs_pred[deps[1]]))

                H_qo = float(-(qo * log_stable(qo)).sum())
                ext += float((qo * C_m).sum())
                epi += amb - H_qo    # (-ve) epistemic = -MI = E_s[H[P(o|s)]] - H[q(o)] ≤ 0

            qs_t = qs_pred

        ext_list.append(ext)
        epi_list.append(epi)

    ext_arr  = jnp.array(ext_list)
    epi_arr  = jnp.array(epi_list)
    extrinsic = float((q_pi_vec * ext_arr).sum())
    epistemic = float((q_pi_vec * epi_arr).sum())
    return extrinsic, epistemic, ext_list, epi_list


@dataclass
class SimConfig:
    trial_schedule: list | None = None
    num_trials: int = 0
    lr_A1: float = 3.0
    lr_B1: float = 3.0
    lr_D1: float = 3.0
    lr_A2: float = 3.0
    lr_D2: float = 3.0
    gamma_lower: float = 3.0
    gamma_higher: float = 3.0
    alpha_lower: float = 6.0
    a0_A1: float = 100.0
    b0_B1: float = 100.0
    d0_D1: float = 100.0
    a0_A2: float = 16.0
    d0_D2: float = 16.0
    num_iter: int = 8
    linkC_social: Any = None   # (2,2) ndarray or None → default 사용
    linkC_heart: Any = None    # (2,2) ndarray or None → default 사용
    use_topdown: bool = True    # False → top-down D/C 적용 안 함
    policy_len: int = 3         # planning horizon sweep
    a1_val: float = 0.70        # A1_heart = [[a, 1-a], [1-a, a]]; 0.5=flat, 1.0=deterministic
    c1_val: float = 0.0         # C1_heart = [-c, c]; 0=no preference
    a2_init: Any = None         # 초기 A2 행렬 override
    d2_init: Any = None         # 초기 D2 벡터 override

    def __post_init__(self):
        # JAX 배열을 numpy로 변환 — multiprocessing pickle 안전성 보장
        for attr in ('a2_init', 'd2_init', 'linkC_social', 'linkC_heart'):
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, np.asarray(val))

class TwoLayerSocialEnv:
    """
    s2: passive / active
    s1_social: unattended / attended
    s1_heart: low / high
    o1_social: no_eye_contact / eye_contact
    o1_heart: low / high
    """

    def __init__(self):
        # P(s_social1 | s2)
        self.link_true_social = jnp.array([
            [0.95, 0.05],  # unattended | passive, active
            [0.05, 0.95],  # attended   | passive, active
        ])
        # P(s_heart1 | s2)
        self.link_true_heart = jnp.array([
            [0.95, 0.05],  # low  | passive, active
            [0.05, 0.95],  # high | passive, active
        ])

        # A1_social: P(face | social_state, heart_state)  [o_social, s_social, s_heart]
        self.A1_social_true = jnp.array([
            # no_eye_contact
            [[0.80, 0.50],  # unattended: [low, high]
            [0.50, 0.20]],   # attended:   [low, high]
            # eye_contact
            [[0.20, 0.50],  # unattended: [low, high]
            [0.50, 0.80]]   # attended:   [low, high]
            ])

        # A1_heart: P(heart_obs | heart_state)
        self.A1_heart_true = jnp.array([
            [0.70, 0.30],  # o = low  | s_heart = low, high
            [0.30, 0.70],  # o = high
            ])

        # B1_social(action)
        self.B1_social_true = jnp.stack([
            # u = silent
            jnp.array([
                [0.90, 0.70],  # unattended_t | unattended, attended
                [0.10, 0.30],  # attended_t
            ]),
            # u = answer
            jnp.array([
                [0.30, 0.10],  # unattended_t | unattended, attended
                [0.70, 0.90],  # attended_t
            ]),
        ], axis=-1)

        # B1_heart(next_heart, prev_heart, prev_social, action)
        silent_true = jnp.array([
            # next = low
            [[0.95, 0.65],   # prev_heart = low ; prev_social = unattended, attended
            [0.65, 0.35]],  # prev_heart = high

            # next = high
            [[0.05, 0.35],
            [0.35, 0.65]],
            ])
        answer_true = jnp.array([
            # next = low
            [[0.65, 0.35],
            [0.35, 0.05]],

            # next = high
            [[0.35, 0.65],
            [0.65, 0.95]],
            ])
        self.B1_heart_true = jnp.stack([silent_true, answer_true], axis=-1)

    # s2 샘플링 (사용 X)
    def sample_true_s2(self, rng_key: jnp.ndarray) -> int:
        probs = jnp.array([0.20, 0.80])
        return sample_categorical(rng_key, probs)

    # true s2가 주어졌을 때 그에 맞는 s1를 샘플링 (사용 X)
    def sample_initial_s1(self, s2_true: int, rng_social: jnp.ndarray, rng_heart: jnp.ndarray) -> tuple[int, int]:
        social = sample_categorical(rng_social, self.link_true_social[:, s2_true])
        heart = sample_categorical(rng_heart, self.link_true_heart[:, s2_true])
        return social, heart

    # 주어진 hidden states로부터 실제 observation 생성
    def sample_obs(self, s_social: int, s_heart: int, rng_social: jnp.ndarray, rng_heart: jnp.ndarray) -> tuple[int, int]:
        o_social = sample_categorical(rng_social, self.A1_social_true[:, s_social, s_heart])
        o_heart = sample_categorical(rng_heart, self.A1_heart_true[:, s_heart])
        return o_social, o_heart

    # 현재 state과 action으로 다음 state 샘플링
    def step_transition(self, s_social: int, s_heart: int, action: int, rng_social: jnp.ndarray, rng_heart: jnp.ndarray) -> tuple[int, int]:
        nsocial_social = sample_categorical(rng_social, self.B1_social_true[:, s_social, action])
        nsocial_heart = sample_categorical(rng_heart, self.B1_heart_true[:, s_heart, s_social, action])
        return nsocial_social, nsocial_heart


def build_lower_agent(cfg: SimConfig, env: TwoLayerSocialEnv, key: jnp.ndarray) -> Agent:
    policies = Policies(jnp.array(
        [[[0, 0]] * cfg.policy_len, [[1, 1]] * cfg.policy_len],
        dtype=jnp.int32,
    ))

    A1_social = np.array(env.A1_social_true)
    a = cfg.a1_val
    A1_heart = np.array([[a, 1 - a],
                          [1 - a, a]])

    # pA : A의 Dirichlet concentration (a)
    pA1_social = A1_social * cfg.a0_A1
    pA1_heart = A1_heart * cfg.a0_A1

    B1_social = np.array(env.B1_social_true)
    B1_heart = np.array(env.B1_heart_true)

    pB1_social = B1_social * cfg.b0_B1
    pB1_heart = B1_heart * cfg.b0_B1

    D1_social = np.array([0.5, 0.5])    # unattended, attended
    D1_heart = np.array([0.5, 0.5])   # low, high

    pD1_social = D1_social * cfg.d0_D1
    pD1_heart = D1_heart * cfg.d0_D1

    C1_social = np.array([2.0, 2.0])
    C1_heart = np.array([-cfg.c1_val, cfg.c1_val])

    lower = Agent(
        A=[A1_social, A1_heart],
        B=[B1_social, B1_heart],
        C=[C1_social, C1_heart],
        D=[D1_social, D1_heart],
        E=np.ones((1, policies.num_policies)) / policies.num_policies,
        pA=[pA1_social, pA1_heart],
        pB=[pB1_social, pB1_heart],
        pD=[pD1_social, pD1_heart],
        A_dependencies=[[0, 1], [1]],
        B_dependencies=[[0], [1, 0]],
        num_controls=[2, 2],
        policies=policies,
        policy_len=cfg.policy_len,
        gamma=cfg.gamma_lower,
        alpha=cfg.alpha_lower,
        inference_algo='fpi',
        action_selection='stochastic',
        sampling_mode='full',
        learn_A=True,
        learn_B=True,
        learn_D=True,
        categorical_obs=False,
        batch_size=1,
        num_iter=cfg.num_iter,
    )
    return lower


def build_higher_agent(cfg: SimConfig) -> Agent:
    A2 = (np.asarray(cfg.a2_init) if cfg.a2_init is not None else np.array([
        [0.80, 0.20],
        [0.20, 0.80],
    ]))
    pA2 = A2 * cfg.a0_A2

    B2 = np.array([
        [[0.90], [0.10]],
        [[0.10], [0.90]],
    ])

    D2 = (np.asarray(cfg.d2_init) if cfg.d2_init is not None else np.array([0.50, 0.50]))
    pD2 = D2 * cfg.d0_D2

    higher = Agent(
        A=[A2],
        B=[B2],
        C=[np.array([0.0, 0.0])],
        D=[D2],
        E=np.ones((1, 1)),
        policies=Policies(jnp.array([[[0]]], dtype=jnp.int32)),
        pA=[pA2],
        pD=[pD2],
        A_dependencies=[[0]],
        B_dependencies=[[0]],
        num_controls=[1],
        policy_len=1,
        gamma=cfg.gamma_higher,
        alpha=1.0,
        inference_algo='fpi',
        action_selection='stochastic',
        sampling_mode='full',
        learn_A=True,
        learn_B=False,
        learn_D=True,
        categorical_obs=True,
        batch_size=1,
        num_iter=cfg.num_iter,
    )
    return higher


class HierarchicalController:
    def __init__(
        self,
        lower: Agent,
        higher: Agent,
        d_blend: float = 1,
        c_blend: float = 1,
        linkC_social: Any = None,
        linkC_heart: Any = None,
    ):
        self.lower = lower
        self.higher = higher

        self.d_blend = d_blend
        self.c_blend = c_blend

        # lower posterior q(s1) -> higher observation o2
        # row = higher outcome (low_demand/high_demand), col = lower state
        self.link_insula_social = jnp.array([
            [0.80, 0.20],  # low_demand  | unattended, attended
            [0.20, 0.80],  # high_demand
            ])
        self.link_insula_heart = jnp.array([
            [0.80, 0.20],  # low_demand  | low, high
            [0.20, 0.80],  # high_demand
            ])

        # linkD: o2 (2 outcomes) -> D1 (2 states)
        self.linkD_social = jnp.array([
            [0.90, 0.10],  # unattended | low_demand, high_demand
            [0.10, 0.90],  # attended
            ], dtype=jnp.float32)
        self.linkD_heart = jnp.array([
            [0.90, 0.10],  # low  | low_demand, high_demand
            [0.10, 0.90],  # high
        ], dtype=jnp.float32)

        # linkC: higher predicted outcome -> lower C1
        # social preference from higher social predicted outcome
        _default_linkC_social = jnp.array([
            [0.50, -0.50],  # no_eye_contact | low_demand, high_demand
            [-0.50, 0.50],  # eye_contact
        ])
        _default_linkC_heart = jnp.array([
            [0.50, -0.50],  # low  | low_demand, high_demand
            [-0.50, 0.50],  # high
        ])
        # parameter sweep 대비
        self.linkC_social = jnp.asarray(linkC_social) if linkC_social is not None else _default_linkC_social
        self.linkC_heart  = jnp.asarray(linkC_heart)  if linkC_heart  is not None else _default_linkC_heart

    # q(s1) -> o2
    def lower_to_higher_obs(self, qs_lower: list[jnp.ndarray]) -> list[jnp.ndarray]:
        q_social = qs_lower[0][0, -1]
        q_heart  = qs_lower[1][0, -1]

        # link_social/heart: (2,2) → social/heart 각각 계산 후 평균
        o2 = (self.link_insula_social @ q_social + self.link_insula_heart @ q_heart) / 2.0  # (2,)

        return [o2[None, :]]   # [(1, 2)]

    def higher_predicted_outcomes(self, qs_2):
        qs_2 = jnp.asarray(qs_2).reshape(-1)           # (2,)
        A2 = self.higher.A[0]                           # (1, 2, 2) after batch dim 추가
        o2 = (A2 @ qs_2)[0]                             # (1,2,2)@(2,) → (1,2) → [0] → (2,)
        return o2

    # q(s2) -> o2 -> D1
    def higher_to_lower_D(self, qs_2: jnp.ndarray) -> list[jnp.ndarray]:
        o2 = self.higher_predicted_outcomes(qs_2)    # (2,)

        d_social = self.linkD_social @ o2
        d_heart  = self.linkD_heart  @ o2

        return [d_social, d_heart]
    
    def apply_topdown_D(self, D_new: list[jnp.ndarray]) -> None:
        old_D_social = self.lower.D[0][0]
        old_D_heart = self.lower.D[1][0]

        blended_D_social = (1.0 - self.d_blend) * old_D_social + self.d_blend * D_new[0]
        blended_D_heart  = (1.0 - self.d_blend) * old_D_heart  + self.d_blend * D_new[1]

        lower = tree_at(lambda a: a.D[0], self.lower, blended_D_social[None, :])
        self.lower = tree_at(lambda a: a.D[1], lower, blended_D_heart[None, :])

    # q(s2) -> o2 -> C1  (log-pref 직접 반환)
    def higher_to_lower_C(self, qs_2: jnp.ndarray) -> list[jnp.ndarray]:
        o2 = self.higher_predicted_outcomes(qs_2)
        c_social = self.linkC_social @ o2
        c_heart  = self.linkC_heart  @ o2
        return [c_social, c_heart]

    def apply_topdown_C(self, C_new: list[jnp.ndarray]) -> None:
        old_C_social = self.lower.C[0][0]
        old_C_heart  = self.lower.C[1][0]

        blended_C_social = (1.0 - self.c_blend) * old_C_social + self.c_blend * C_new[0]
        blended_C_heart  = (1.0 - self.c_blend) * old_C_heart  + self.c_blend * C_new[1]

        lower = tree_at(lambda a: a.C[0], self.lower, blended_C_social[None, :])
        self.lower = tree_at(lambda a: a.C[1], lower, blended_C_heart[None, :])

    # q(s2) -> A2, D2
    def update_higher_with_current_obs(
        self,
        qs_higher_t0: list[jnp.ndarray],
        obs_higher_t0: list[jnp.ndarray],
        qs_higher_t1: list[jnp.ndarray],
        obs_higher_t1: list[jnp.ndarray],
        cfg: SimConfig
    ) -> None:
        # t0, t1 belief/obs 모두 반영 → (batch=1, T=2, ...)
        higher_qs_seq = stack_beliefs_over_time([qs_higher_t0, qs_higher_t1])
        obs_higher_seq = [
            jnp.concatenate([obs_higher_t0[0][:, None, :], obs_higher_t1[0][:, None, :]], axis=1),  # (1, T=2, No=2)
        ]

        lr_D2_half = cfg.lr_D2 / 2.0

        # A2 + D2(t0) learning — pymdp는 내부적으로 [:, 0]만 D에 반영
        self.higher = self.higher.infer_parameters(
            beliefs_A=[higher_qs_seq[0]],
            observations=obs_higher_seq,
            actions=None,
            beliefs_B=None,
            lr_pA=cfg.lr_A2,
            lr_pD=lr_D2_half,
        )

        # D2(t1) 수동 반영: t0과 동일 비율(lr_D2/2)로 누적
        qs_t1 = qs_higher_t1[0][:, -1, :]          # (1, 2)
        new_pD = self.higher.pD[0] + lr_D2_half * qs_t1
        new_D  = normalize(new_pD)
        self.higher = tree_at(lambda x: (x.D[0], x.pD[0]), self.higher, (new_D, new_pD))


def run_simulation(cfg: SimConfig, seed: int = 7) -> dict[str, Any]:
    rng = jr.PRNGKey(seed)

    rng, k_build = jr.split(rng)
    env = TwoLayerSocialEnv()
    lower = build_lower_agent(cfg, env, k_build)
    higher = build_higher_agent(cfg)
    controller = HierarchicalController(lower, higher,
                                        linkC_social=cfg.linkC_social,
                                        linkC_heart=cfg.linkC_heart)

    # ===== 초기 파라미터 저장 =====
    init_lower_A = [a.copy() for a in controller.lower.A]
    init_lower_B = [b.copy() for b in controller.lower.B]
    init_lower_D = [d.copy() for d in controller.lower.D]

    init_higher_A = [a.copy() for a in controller.higher.A]
    init_higher_D = [d.copy() for d in controller.higher.D]

    # ===== logs dictionary 초기화 =====
    logs: dict[str, list[Any]] = {
        'trial': [],
        'obs_2_t0': [],
        'qs_2_t0': [],
        'obs_2_t1': [],
        'qs_2_t1': [],
        'qs_1_t0': [],
        'qs_1_t1': [],
        'D1_social_td': [],
        'D1_heart_td': [],
        'C1_social_td': [],
        'C1_heart_td': [],
        'action': [],
        'D2': [],
        'A2': [],
        'A1_social': [],
        'A1_heart': [],
        'B1_social': [],
        'B1_heart': [],

        'q_pi_list': [],
        'extrinsic': [],
        'epistemic': [],
        'extrinsic_per_pi': [],
        'epistemic_per_pi': [],
        'vfe_t0': [],
        'vfe_t1': [],
        'G_pi': [],
        'D1_social': [],
        'D1_heart': [],
        'obs_1_t0': [],
        'obs_1_t1': [],
    }

    # ===== 시뮬레이션 실행 =====
    for trial in range(cfg.num_trials):
        rng, k_s2, k_e0, k_h0, k_oe0, k_oh0, k_act, k_e1, k_h1, k_oe1, k_oh1 = jr.split(rng, 11)

        # s2, s1 결정: trial_schedule이 있으면 직접 지정, 없으면 샘플링
        if cfg.trial_schedule is not None:
            s_social0, s_heart0 = cfg.trial_schedule[trial]
            s2_true = -1  # 직접 지정 시 s2 미사용
        else:
            s2_true = env.sample_true_s2(k_s2)
            s_social0, s_heart0 = env.sample_initial_s1(s2_true, k_e0, k_h0)
        o_social0, o_heart0 = env.sample_obs(s_social0, s_heart0, k_oe0, k_oh0)

        # q(s1) inference
        qs_lower_t0 = controller.lower.infer_states(
            [jnp.array([o_social0], dtype=jnp.int32), jnp.array([o_heart0], dtype=jnp.int32)],
            empirical_prior=controller.lower.D,
        )

        # q(s1) -> o2
        higher_obs_t0 = controller.lower_to_higher_obs(qs_lower_t0)

        # q(s2) inference
        qs_higher_t0 = controller.higher.infer_states(
            higher_obs_t0,
            empirical_prior=controller.higher.D,
        )
        # q(s2) -> D1, C1
        D1_topdown = controller.higher_to_lower_D(qs_higher_t0[0][0, -1])
        C1_topdown = controller.higher_to_lower_C(qs_higher_t0[0][0, -1])
        if cfg.use_topdown:
            controller.apply_topdown_D(D1_topdown)
            controller.apply_topdown_C(C1_topdown)
        
        # D1 변경 이후 lower re-inference
        prior_t0 = [d[0] for d in controller.lower.D]
        qs_lower_t0_td = controller.lower.infer_states(
            [jnp.array([o_social0], dtype=jnp.int32), jnp.array([o_heart0], dtype=jnp.int32)],
            empirical_prior=controller.lower.D,
        )
        _, vfe_t0 = calc_vfe(
            [q[0, -1] for q in qs_lower_t0_td], prior_t0,
            obs=[o_social0, o_heart0],
            A=[a[0] for a in controller.lower.A],
            A_dependencies=controller.lower.A_dependencies,
            distr_obs=False,
        )
        vfe_t0 = float(vfe_t0)

        # lower policy inference & action selection
        q_pi, G_pi = controller.lower.infer_policies(qs_lower_t0_td)
        action = controller.lower.sample_action(q_pi, rng_key=k_act[None, ...])
        a_t = int(action[0, 0])

        # action 이후 s1, o1 생성
        s_social1, s_heart1 = env.step_transition(s_social0, s_heart0, a_t, k_e1, k_h1)
        o_social1, o_heart1 = env.sample_obs(s_social1, s_heart1, k_oe1, k_oh1)

        empirical_prior_t1 = controller.lower.update_empirical_prior(action, qs_lower_t0_td)
        qs_lower_t1 = controller.lower.infer_states(
            [jnp.array([o_social1], dtype=jnp.int32), jnp.array([o_heart1], dtype=jnp.int32)],
            empirical_prior=empirical_prior_t1,
        )
        prior_t1 = [p[0] if p.ndim == 2 else p[0, -1] for p in empirical_prior_t1]
        _, vfe_t1 = calc_vfe(
            [q[0, -1] for q in qs_lower_t1], prior_t1,
            obs=[o_social1, o_heart1],
            A=[a[0] for a in controller.lower.A],
            A_dependencies=controller.lower.A_dependencies,
            distr_obs=False,
        )
        vfe_t1 = float(vfe_t1)
    
        # q(s1_t1) -> o2 -> q(s2_t1) inference (시간 연속성: B2 · q(s2_t0))
        higher_obs_t1 = controller.lower_to_higher_obs(qs_lower_t1)
        action_higher = jnp.zeros((1, 1), dtype=jnp.int32)  # higher는 실제 action 없음
        empirical_prior_higher_t1 = controller.higher.update_empirical_prior(action_higher, qs_higher_t0)
        qs_higher_t1 = controller.higher.infer_states(
            higher_obs_t1,
            empirical_prior=empirical_prior_higher_t1,
        )
        # A2, D2 learning (t0, t1 둘 다 반영)
        controller.update_higher_with_current_obs(qs_higher_t0, higher_obs_t0, qs_higher_t1, higher_obs_t1, cfg)

        # A1, B1 learning (t0, t1 둘 다 반영)
        lower_qs_seq = stack_beliefs_over_time([qs_lower_t0_td, qs_lower_t1])
        lower_obs_seq = [
            jnp.array([[o_social0, o_social1]], dtype=jnp.int32),
            jnp.array([[o_heart0, o_heart1]], dtype=jnp.int32),
        ]
        lower_actions_seq = jnp.array([[[a_t, a_t]]], dtype=jnp.int32)
        controller.lower = controller.lower.infer_parameters(
            beliefs_A=lower_qs_seq,
            beliefs_B=lower_qs_seq,
            observations=lower_obs_seq,
            actions=lower_actions_seq,
            lr_pA=cfg.lr_A1,
            lr_pB=cfg.lr_B1,
            lr_pD=cfg.lr_D1,
        )


        # ===== 결과 지표 저장 =====
        d1_social = controller.lower.D[0][0]   # (2,) top-down prior: unattended, attended
        d1_heart  = controller.lower.D[1][0]   # (2,) top-down prior: low, high
        extrinsic, epistemic, ext_per_pi, epi_per_pi = compute_EFE_components(controller.lower, qs_lower_t0_td, q_pi)

        logs['D2'].append(np.array(controller.higher.D[0][0]))
        logs['A2'].append(np.array(controller.higher.A[0][0]))
        logs["q_pi_list"].append(np.array(q_pi))
        logs['trial'].append(trial)
        logs['obs_2_t0'].append(np.array(higher_obs_t0[0]))
        logs['obs_2_t1'].append(np.array(higher_obs_t1[0]))
        logs['qs_2_t1'].append(np.array(qs_higher_t1[0][0, -1]))
        logs['qs_2_t0'].append(np.array(qs_higher_t0[0][0, -1]))
        logs['qs_1_t0'].append(np.stack([np.array(q[0, -1]) for q in qs_lower_t0_td]))
        logs['qs_1_t1'].append(np.stack([np.array(q[0, -1]) for q in qs_lower_t1]))
        logs['D1_social_td'].append(np.array(D1_topdown[0]))
        logs['D1_heart_td'].append(np.array(D1_topdown[1]))
        logs['C1_social_td'].append(np.array(C1_topdown[0]))
        logs['C1_heart_td'].append(np.array(C1_topdown[1]))
        logs['action'].append(int(a_t))
        logs['extrinsic'].append(extrinsic)
        logs['epistemic'].append(epistemic)
        logs['extrinsic_per_pi'].append(np.array(ext_per_pi))
        logs['epistemic_per_pi'].append(np.array(epi_per_pi))
        logs['vfe_t0'].append(vfe_t0)
        logs['vfe_t1'].append(vfe_t1)
        logs['G_pi'].append(np.array(G_pi))
        logs['D1_social'].append(np.array(d1_social))
        logs['D1_heart'].append(np.array(d1_heart))
        logs['obs_1_t0'].append(np.array([o_social0, o_heart0]))
        logs['obs_1_t1'].append(np.array([o_social1, o_heart1]))
        logs['A1_social'].append(np.array(controller.lower.A[0][0]))
        logs['A1_heart'].append(np.array(controller.lower.A[1][0]))
        logs['B1_social'].append(np.array(controller.lower.B[0][0]))
        logs['B1_heart'].append(np.array(controller.lower.B[1][0]))

    return {
        "logs": logs,
        "lower_A": [np.array(a[0]) for a in controller.lower.A],
        "lower_B": [np.array(b[0]) for b in controller.lower.B],
        "lower_D": [np.array(d[0]) for d in controller.lower.D],
        "higher_A": [np.array(a[0]) for a in controller.higher.A],
        "higher_D": [np.array(d[0]) for d in controller.higher.D],
        "init_lower_A": [np.array(a) for a in init_lower_A],
        "init_lower_B": [np.array(b) for b in init_lower_B],
        "init_lower_D": [np.array(d) for d in init_lower_D],
        "init_higher_A": [np.array(a) for a in init_higher_A],
        "init_higher_D": [np.array(d) for d in init_higher_D],
    }


_SCALAR_KEYS = ['extrinsic', 'epistemic', 'vfe_t0', 'vfe_t1']
_ARRAY_KEYS = [
    'q_pi_list', 'G_pi',
    'D1_social', 'D1_heart',
    'obs_2_t0', 'obs_2_t1',
    'qs_2_t0', 'qs_2_t1',
    'qs_1_t0', 'qs_1_t1',
    'D1_social_td', 'D1_heart_td',
    'C1_social_td', 'C1_heart_td',
    'D2', 'A2',
    'A1_social', 'A1_heart',
    'B1_social', 'B1_heart',
    'obs_1_t0', 'obs_1_t1',
    'extrinsic_per_pi', 'epistemic_per_pi',
]


def average_logs(all_seed_logs: list[dict]) -> dict:
    """여러 seed의 logs를 trial별로 평균."""
    n_trials = len(all_seed_logs[0]['trial'])
    avg: dict[str, Any] = {}

    avg['trial'] = all_seed_logs[0]['trial']

    # action: 0/1 정수 → seed별 평균 비율(float)
    avg['action'] = list(
        np.mean([[float(lg['action'][t]) for t in range(n_trials)] for lg in all_seed_logs], axis=0)
    )

    for key in _SCALAR_KEYS:
        avg[key] = list(
            np.mean([[float(lg[key][t]) for t in range(n_trials)] for lg in all_seed_logs], axis=0)
        )

    for key in _ARRAY_KEYS:
        # (n_seeds, n_trials, ...) → mean over seeds → list of per-trial arrays
        stacked = np.stack([
            np.stack([np.asarray(lg[key][t]) for t in range(n_trials)])
            for lg in all_seed_logs
        ])  # (n_seeds, n_trials, ...)
        means = stacked.mean(axis=0)  # (n_trials, ...)
        avg[key] = [jnp.asarray(means[t]) for t in range(n_trials)]

    return avg


def _run_simulation_wrapped(cfg: SimConfig, seed: int) -> dict:
    import os, traceback as _tb
    # Windows spawn 방식에서 각 worker가 JAX를 재초기화할 때 GPU 충돌 방지
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        return run_simulation(cfg, seed)
    except Exception as exc:
        raise RuntimeError(f"seed={seed} failed:\n{_tb.format_exc()}") from None


def run_multi_seed(cfg: SimConfig, seeds: list[int], max_workers: int | None = None) -> dict[str, Any]:
    """여러 seed로 시뮬레이션을 실행하고 결과를 평균."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    n_workers = max_workers if max_workers is not None else min(len(seeds), os.cpu_count() or 1)
    all_results: list[Any] = [None] * len(seeds)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(_run_simulation_wrapped, cfg, seed): i
                         for i, seed in enumerate(seeds)}
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            all_results[i] = future.result()
            print(f"  [seed={seeds[i]}] done", flush=True)

    avg_logs = average_logs([r["logs"] for r in all_results])
    last = all_results[-1]

    n_lower_A = len(last["lower_A"])
    n_lower_B = len(last["lower_B"])
    n_lower_D = len(last["lower_D"])
    n_higher_A = len(last["higher_A"])
    n_higher_D = len(last["higher_D"])

    avg_lower_A = [
        jnp.mean(jnp.stack([r["lower_A"][m] for r in all_results]), axis=0)
        for m in range(n_lower_A)
    ]
    avg_lower_B = [
        jnp.mean(jnp.stack([r["lower_B"][f] for r in all_results]), axis=0)
        for f in range(n_lower_B)
    ]
    avg_lower_D = [
        jnp.mean(jnp.stack([r["lower_D"][f] for r in all_results]), axis=0)
        for f in range(n_lower_D)
    ]
    avg_higher_A = [
        jnp.mean(jnp.stack([r["higher_A"][m] for r in all_results]), axis=0)
        for m in range(n_higher_A)
    ]
    avg_higher_D = [
        jnp.mean(jnp.stack([r["higher_D"][f] for r in all_results]), axis=0)
        for f in range(n_higher_D)
    ]

    return {
        "logs": avg_logs,
        "seed_logs": [r["logs"] for r in all_results],
        "init_lower_A": last["init_lower_A"],
        "init_lower_B": last["init_lower_B"],
        "init_lower_D": last["init_lower_D"],
        "init_higher_A": last["init_higher_A"],
        "init_higher_D": last["init_higher_D"],
        "avg_lower_A": avg_lower_A,
        "avg_lower_B": avg_lower_B,
        "avg_lower_D": avg_lower_D,
        "avg_higher_A": avg_higher_A,
        "avg_higher_D": avg_higher_D,
    }


def make_linkC_value(w_social: float, w_heart: float):
    """
    value-based mapping (NOT probability)

    positive = 선호
    negative = 회피
    """

    linkC_social = np.array([
        [ w_social, -w_social],   # no_eye_contact
        [-w_social,  w_social],   # eye_contact
    ])

    linkC_heart = np.array([
        [ w_heart, -w_heart],     # low
        [-w_heart,  w_heart],     # high
    ])

    return linkC_social, linkC_heart


def print_result_table(logs, label: str):
    ACT = ['silent', 'answer']
    HDR = (f"{'trial':>5} | {'act(rate)':^9} | {'q_pi[sil,ans]':^22} | {'G[sil,ans]':^22} | "
           f"{'H(q(pi))':>10} | {'ext[sil,ans]':^26} | {'epi[sil,ans]':^26}")
    SEP = "=" * len(HDR)
    print(f"\n{SEP}")
    print(f"  Condition: {label}")
    print(SEP)
    print(HDR)
    print("-" * len(HDR))
    for i in range(len(logs['trial'])):
        q     = logs['q_pi_list'][i]
        q_s   = f"[{float(q[0,0]):5.3f}, {float(q[0,1]):5.3f}]"
        G     = logs['G_pi'][i]
        G_s   = f"[{float(G[0,0]):+6.3f}, {float(G[0,1]):+6.3f}]"
        H_ppi = entropy(q[0])

        act_val = logs['action'][i]
        if isinstance(act_val, int) or (isinstance(act_val, float) and act_val in (0.0, 1.0)):
            act_s = ACT[int(act_val)]
        else:
            act_s = f"{act_val:.3f}"

        ext = logs['extrinsic_per_pi'][i]
        epi = logs['epistemic_per_pi'][i]
        ext_s = f"[{float(ext[0]):+7.3f}, {float(ext[1]):+7.3f}]"
        epi_s = f"[{float(epi[0]):+7.3f}, {float(epi[1]):+7.3f}]"

        print(f"{i+1:>5} | {act_s:^9} | {q_s:^22} | {G_s:^22} | "
              f"{H_ppi:>10.3f} | {ext_s:^26} | {epi_s:^26}")
    print(SEP)


if __name__ == '__main__':
    # 빠른 동작 확인용
    linkC_s, linkC_h = make_linkC_value(-0.90, -0.90)
    cfg = SimConfig(
        num_trials=25,
        trial_schedule=[(1, 1)] * 25,
        linkC_social=linkC_s,
        linkC_heart=linkC_h,
    )
    result = run_multi_seed(cfg, seeds=[3, 4, 5])
    print_result_table(result["logs"], "sanity check")