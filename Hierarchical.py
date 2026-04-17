from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
import jax.random as jr
from equinox import tree_at
from pymdp.agent import Agent
from pymdp.maths import calc_vfe

key = jr.PRNGKey(0)



def normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-16) -> jnp.ndarray:
    x = jnp.asarray(x)
    x = jnp.clip(x, a_min=eps) # log0, 0/0 방지
    return x / x.sum(axis=axis, keepdims=True)


def log_stable(x: jnp.ndarray, eps: float = 1e-16) -> jnp.ndarray:
    return jnp.log(jnp.clip(x, a_min=eps))


def entropy(p: jnp.ndarray, eps: float = 1e-16) -> float:
    return float(-jnp.sum(p * log_stable(p, eps=eps)))


def sample_categorical(rng_key: jnp.ndarray, probs: jnp.ndarray) -> int:
    probs = normalize(probs)
    return int(jr.categorical(rng_key, jnp.log(probs))) # 랜덤 샘플링으로 얻은 index


# A1, B1 learning
def stack_beliefs_over_time(qs_list: list[list[jnp.ndarray]]) -> list[jnp.ndarray]:
    num_factors = len(qs_list[0])
    out: list[jnp.ndarray] = []
    for f in range(num_factors):
        out.append(jnp.concatenate([qs_t[f] for qs_t in qs_list], axis=1))
    return out


# print 용

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
    return extrinsic, epistemic




@dataclass # init 역할까지 해줌
class SimConfig:
    trial_schedule: list | None = None
    num_trials: int = 0
    lr_A1: float = 3.0
    lr_B1: float = 3.0
    lr_A2: float = 3.0
    lr_D2: float = 3.0
    gamma_lower: float = 3.0
    gamma_higher: float = 3.0
    alpha_lower: float = 3.0
    a0: float = 16.0
    b0: float = 16.0
    d0: float = 16.0
    num_iter: int = 8
    linkC_social: Any = None   # (2,2) jnp.ndarray or None → default 사용
    linkC_heart: Any = None    # (2,2) jnp.ndarray or None → default 사용

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
            # o_social = no_eye_contact
            [[0.60, 0.45],  # s_social=unattended; s_heart=low, high
             [0.40, 0.30]], # s_social=attended
            # o_social = eye_contact
            [[0.40, 0.55],
             [0.60, 0.70]],
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
            [[0.95, 0.70],   # prev_heart=low;  prev_social=unattended, attended
             [0.80, 0.55]],  # prev_heart=high
            # next = high
            [[0.05, 0.30],
             [0.20, 0.45]],
        ])
        answer_true = jnp.array([
            # next = low
            [[0.45, 0.30],   # prev_heart=low;  prev_social=unattended, attended
             [0.20, 0.05]],  # prev_heart=high
            # next = high
            [[0.55, 0.70],
             [0.80, 0.95]],
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
    # planning_horizon = 3
    policies = jnp.array([
        [[0, 0], [0, 0], [0, 0]],  # silent, silent, silent
        [[1, 1], [1, 1], [1, 1]],  # answer, answer, answer
    ], dtype=jnp.int32)

    # # planning_horizon = 1
    # policies = jnp.array([
    #     [[0, 0]],  # silent
    #     [[1, 1]],  # answer
    # ], dtype=jnp.int32)

    # A1_social(face | social, heart)  [o_social, s_social, s_heart]
    A1_social, A1_heart = env.A1_social_true, env.A1_heart_true

    # pA : A의 Dirichlet concentration (a)
    pA1_social = A1_social * cfg.a0
    pA1_heart = A1_heart * cfg.a0

    B1_social, B1_heart = env.B1_social_true, env.B1_heart_true

    pB1_social = B1_social * cfg.b0
    pB1_heart = B1_heart * cfg.b0

    D1_social = jnp.array([0.5, 0.5])    # unattended, attended
    D1_heart = jnp.array([0.5, 0.5])   # low, high

    C1_social = jnp.array([0.0, 0.0])    # no_eye_contact, eye_contact (log space)
    C1_heart = jnp.array([0.0, 0.0])    # low 선호 (log space)

    lower = Agent(
        A=[A1_social, A1_heart],
        B=[B1_social, B1_heart],
        C=[C1_social, C1_heart],
        D=[D1_social, D1_heart],
        E=jnp.ones((1, policies.shape[0])) / policies.shape[0],
        pA=[pA1_social, pA1_heart],
        pB=[pB1_social, pB1_heart],
        A_dependencies=[[0, 1], [1]],
        B_dependencies=[[0], [1, 0]],
        num_controls=[2, 2],
        policies=policies,
        policy_len=3, #planning_horizon = 3
        gamma=cfg.gamma_lower,
        inference_algo='fpi',
        action_selection='deterministic',
        sampling_mode='full',
        learn_A=True,
        learn_B=True,
        learn_D=False,
        categorical_obs=False,
        batch_size=1,
        num_iter=cfg.num_iter,
    )
    return lower


def build_higher_agent(cfg: SimConfig) -> Agent:
    # shape (num_obs=2, num_states=2): pymdp가 batch dim 자동 추가 → (1,2,2)
    A2 = jnp.array([
        [0.82, 0.18],  # P(obs=low_demand  | s2=passive, active)
        [0.18, 0.82],  # P(obs=high_demand | s2=passive, active)
    ])
    pA2 = A2 * cfg.a0

    B2 = jnp.array([
        [[0.88], [0.12]],  # passive → passive 유지
        [[0.12], [0.88]],  # active  → active  유지
    ])

    D2 = jnp.array([0.50, 0.50])
    pD2 = D2 * cfg.d0

    higher = Agent(
        A=[A2],
        B=[B2],
        C=[jnp.array([0.0, 0.0])],   # 1 modality, 2 outcomes
        D=[D2],
        E=jnp.ones((1, 1)),
        pA=[pA2],
        pD=[pD2],
        A_dependencies=[[0]],         # 1 modality
        B_dependencies=[[0]],
        num_controls=[1],
        policy_len=1,
        gamma=cfg.gamma_higher,
        alpha=1.0,
        inference_algo='fpi',
        action_selection='deterministic',
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
            [0.82, 0.18],  # low_demand  | unattended, attended
            [0.18, 0.82],  # high_demand
            ])
        self.link_insula_heart = jnp.array([
            [0.82, 0.18],  # low_demand  | low, high
            [0.18, 0.82],  # high_demand
            ])

        # linkD: o2 (2 outcomes) -> D1 (2 states)
        self.linkD_social = jnp.array([
            [0.82, 0.18],  # unattended | low_demand, high_demand
            [0.18, 0.82],  # attended
            ], dtype=jnp.float32)
        self.linkD_heart = jnp.array([
            [0.82, 0.18],  # low  | low_demand, high_demand
            [0.18, 0.82],  # high
        ], dtype=jnp.float32)

        # linkC: higher predicted outcome -> lower C1
        # social preference from higher social predicted outcome
        _default_linkC_social = jnp.array([
            [0.70, 0.30],  # no_eye_contact | low_demand, high_demand
            [0.30, 0.70],  # eye_contact
        ])
        _default_linkC_heart = jnp.array([
            [0.82, 0.18],  # low  | low_demand, high_demand
            [0.18, 0.82],  # high
        ])
        # parameter sweep 대비
        self.linkC_social = jnp.asarray(linkC_social) if linkC_social is not None else _default_linkC_social
        self.linkC_heart  = jnp.asarray(linkC_heart)  if linkC_heart  is not None else _default_linkC_heart

    # q(s1) -> o2
    def lower_to_higher_obs(self, qs_lower: list[jnp.ndarray]) -> list[jnp.ndarray]:
        q_social = normalize(qs_lower[0][0, -1])    # (2,)
        q_heart = normalize(qs_lower[1][0, -1])  # (2,)

        # link_social/heart: (2,2) → social/heart 각각 계산 후 평균
        o2 = normalize((self.link_insula_social @ q_social + self.link_insula_heart @ q_heart) / 2.0)  # (2,)

        return [o2[None, :]]   # [(1, 2)]

    def higher_predicted_outcomes(self, qs_2):
        qs_2 = jnp.asarray(qs_2).reshape(-1)           # (2,)
        A2 = self.higher.A[0]                           # (1, 2, 2) after batch dim 추가
        o2_raw = (A2 @ qs_2)[0]                         # (1,2,2)@(2,) → (1,2) → [0] → (2,)
        o2 = normalize(o2_raw)
        return o2

    # q(s2) -> o2 -> D1
    def higher_to_lower_D(self, qs_2: jnp.ndarray) -> list[jnp.ndarray]:
        o2 = self.higher_predicted_outcomes(qs_2)    # (2,)

        d_social = normalize(self.linkD_social @ o2)       # (2,2)@(2,) → (2,)
        d_heart = normalize(self.linkD_heart @ o2)   # (2,2)@(2,) → (2,)

        return [d_social, d_heart]
    
    def apply_topdown_D(self, D_new: list[jnp.ndarray]) -> None:
        old_D_social = self.lower.D[0][0]
        old_D_heart = self.lower.D[1][0]

        new_D_social = normalize(D_new[0])
        new_D_heart = normalize(D_new[1])

        blended_D_social = normalize((1.0 - self.d_blend) * old_D_social + self.d_blend * new_D_social)
        blended_D_heart = normalize((1.0 - self.d_blend) * old_D_heart + self.d_blend * new_D_heart)

        lower = tree_at(lambda a: a.D[0], self.lower, blended_D_social[None, :])
        self.lower = tree_at(lambda a: a.D[1], lower, blended_D_heart[None, :])

    # q(s2) -> o2 -> C1
    def higher_to_lower_C(self, qs_2: jnp.ndarray) -> list[jnp.ndarray]:
        o2 = self.higher_predicted_outcomes(qs_2)
    
        c_social = normalize(self.linkC_social @ o2)
        c_heart = normalize(self.linkC_heart @ o2)
        return [c_social, c_heart]

    def apply_topdown_C(self, C_new: list[jnp.ndarray]) -> None:
        old_C_social = self.lower.C[0][0]
        old_C_heart = self.lower.C[1][0]

        old_C_social_prob = normalize(jnp.exp(old_C_social))
        old_C_heart_prob = normalize(jnp.exp(old_C_heart))

        new_C_social_prob = C_new[0]
        new_C_heart_prob = C_new[1]

        blended_C_social = log_stable(
            normalize((1.0 - self.c_blend) * old_C_social_prob + self.c_blend * new_C_social_prob)
        )
        blended_C_heart = log_stable(
            normalize((1.0 - self.c_blend) * old_C_heart_prob + self.c_blend * new_C_heart_prob)
        )

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

        self.higher = self.higher.infer_parameters(
            beliefs_A=[higher_qs_seq[0]],
            observations=obs_higher_seq,
            actions=None,
            beliefs_B=None,
            lr_pA=cfg.lr_A2,
            lr_pD=cfg.lr_D2,
        )


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
    
    init_higher_A = [a.copy() for a in controller.higher.A]
    init_higher_D = [d.copy() for d in controller.higher.D]

    # ===== logs dictionary 초기화 =====
    logs: dict[str, list[Any]] = {
        'trial': [],
        'true_s2': [],
        'true_social_t0': [],
        'true_heart_t0': [],
        'obs_t0': [],
        'higher_obs_t0': [],
        'qs_2_t0': [],
        'higher_obs_t1': [],
        'qs_2_t1': [],
        'D1_social_topdown': [],
        'D1_heart_topdown': [],
        'C1_social_topdown': [],
        'C1_heart_topdown': [],
        'q_pi': [],
        'action': [],
        'D2': [],

        'q_pi_list' : [],
        'extrinsic': [],
        'epistemic': [],
        'vfe_t0': [],
        'vfe_t1': [],
        'G_pi': [],
        'd1_social': [],
        'd1_heart': [],
        'lik_social': [],
        'lik_heart': [],
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
        )


        # ===== 결과 지표 저장 =====
        # top-down D1 vs marginal likelihood (A·o) 출력
        d1_social = controller.lower.D[0][0]   # (2,) top-down prior: unattended, attended
        d1_heart  = controller.lower.D[1][0]   # (2,) top-down prior: low, high
        # marginal likelihood: P(o|s) = A[:,s_obs] → likelihood per state
        # social: A1_social marginalised over heart states
        A_soc  = controller.lower.A[0][0]   # (2, 2, 2) = (o_soc, s_soc, s_heart)
        A_hrt  = controller.lower.A[1][0]   # (2, 2)    = (o_hrt, s_heart)
        # likelihood for each s_social: sum over s_heart (uniform marginal)
        lik_social = A_soc[o_social0, :, :].mean(axis=-1)   # (2,) = P(o_soc|s_soc)
        lik_heart  = A_hrt[o_heart0, :]                      # (2,) = P(o_hrt|s_heart)
        extrinsic, epistemic = compute_EFE_components(controller.lower, qs_lower_t0_td, q_pi)

        logs['D2'].append(controller.higher.D[0][0].copy())
        logs["q_pi_list"].append(q_pi.copy())
        logs['trial'].append(trial)
        logs['higher_obs_t0'].append(higher_obs_t0[0])
        logs['higher_obs_t1'].append(higher_obs_t1[0])
        logs['qs_2_t1'].append(qs_higher_t1[0][0, -1])
        logs['qs_2_t0'].append(qs_higher_t0[0][0, -1])
        logs['D1_social_topdown'].append(D1_topdown[0])
        logs['D1_heart_topdown'].append(D1_topdown[1])
        logs['C1_social_topdown'].append(C1_topdown[0])
        logs['C1_heart_topdown'].append(C1_topdown[1])
        logs['action'].append(int(a_t))
        logs['extrinsic'].append(extrinsic)
        logs['epistemic'].append(epistemic)
        logs['vfe_t0'].append(vfe_t0)
        logs['vfe_t1'].append(vfe_t1)
        logs['G_pi'].append(G_pi.copy())
        logs['d1_social'].append(d1_social)
        logs['d1_heart'].append(d1_heart)
        logs['lik_social'].append(lik_social)
        logs['lik_heart'].append(lik_heart)
        logs['final_lower_agent'] = controller.lower
        logs['final_higher_agent'] = controller.higher
    return {
        "logs": logs,
        "controller": controller,
        "init_lower_A": init_lower_A,
        "init_lower_B": init_lower_B,
        "init_higher_A": init_higher_A,
        "init_higher_D": init_higher_D,

    }


def make_linkC(p_social: float, p_heart: float):
    """
        [[p,   1-p],   # no_eye_contact (or low) | low_demand, high_demand
         [1-p, p  ]]   # eye_contact    (or high)
    """
    linkC_social = jnp.array([[p_social, 1 - p_social],
                               [1 - p_social, p_social]])
    linkC_heart  = jnp.array([[p_heart,  1 - p_heart ],
                               [1 - p_heart,  p_heart ]])
    return linkC_social, linkC_heart


def print_result_table(logs, label: str):
    ACT = ['silent', 'answer']
    # HDR = (f"{'trial':>5} | {'action':^7} | {'extrinsic':>10} | {'epistemic':>10} | "
    #        f"{'q_pi[sil,ans]':^22} | "
    #        f"{'G[sil,ans]':^22} | "
    #        f"{'D1_soc[un,at]':^20} | {'D1_hrt[lo,hi]':^20} | "
    #        f"{'Ao_soc[un,at]':^20} | {'Ao_hrt[lo,hi]':^20}")
    HDR = (f"{'trial':>5} | {'action':^7} | {'q_pi[sil,ans]':^22} | {'G[sil,ans]':^22} | "
           f"{'H(q(pi))':>10} | {'VFE(t0)':>9} | {'VFE(t1)':>9}")
    SEP = "=" * 20
    print(f"\n{SEP}")
    print(f"  Condition: {label}")
    print(SEP)
    print(HDR)
    print("-" * 20)
    for i in range(len(logs['trial'])):
        q     = logs['q_pi_list'][i]
        q_s   = f"[{float(q[0,0]):5.3f}, {float(q[0,1]):5.3f}]"
        G     = logs['G_pi'][i]
        G_s   = f"[{float(G[0,0]):+6.3f}, {float(G[0,1]):+6.3f}]"
        d1s   = logs['d1_social'][i]
        d1h   = logs['d1_heart'][i]
        ls    = logs['lik_social'][i]
        lh    = logs['lik_heart'][i]
        d1s_s = f"[{float(d1s[0]):5.3f}, {float(d1s[1]):5.3f}]"
        d1h_s = f"[{float(d1h[0]):5.3f}, {float(d1h[1]):5.3f}]"
        ls_s  = f"[{float(ls[0]):5.3f}, {float(ls[1]):5.3f}]"
        lh_s  = f"[{float(lh[0]):5.3f}, {float(lh[1]):5.3f}]"
        H_ppi  = entropy(q[0])
        # print(f"{i:>5} | {ACT[logs['action'][i]]:^7} | "
        #       f"{logs['extrinsic'][i]:>+10.4f} | {logs['epistemic'][i]:>+10.4f} | "
        #       f"{q_s:^22} | "
        #       f"{G_s:^22} | "
        #       f"{d1s_s:^20} | {d1h_s:^20} | "
        #       f"{ls_s:^20} | {lh_s:^20}")
        print(f"{i+1:>5} | {ACT[logs['action'][i]]:^7} | {q_s:^22} | {G_s:^22} | "
              f"{H_ppi:>10.3f} | {logs['vfe_t0'][i]:>+9.4f} | {logs['vfe_t1'][i]:>+9.4f}")
    print(SEP)


if __name__ == '__main__':
    NUM_TRIALS = 25
    SEED       = 7

    # ===== linkC 조건 정의 =====
    # 각 튜플: (조건 이름, p_social, p_heart)

    # # normal
    # sweep_conditions = [
    #      ("normal",              0.70, 0.82),
    #  ]

    # linkC sweep
    sweep_conditions = [
       ("0.05, 0.82",               0.05, 0.82),
       ("0.10, 0.82",               0.10, 0.82),
       ("0.25, 0.82",               0.25, 0.82),
       ("0.40, 0.82",               0.40, 0.82),
       ("0.50, 0.82",               0.50, 0.82),
       ("0.60, 0.82",               0.60, 0.82),
       ("0.75, 0.82",               0.75, 0.82),
       ("0.90, 0.82",               0.90, 0.82),
       ("0.95, 0.82",               0.95, 0.82),
    ]



    # 시뮬레이션 결과 저장
    all_results = {}

    for cond_name, p_social, p_heart in sweep_conditions:
        linkC_s, linkC_h = make_linkC(p_social, p_heart)

        cfg = SimConfig(num_trials=NUM_TRIALS,
                        linkC_social=linkC_s,
                        linkC_heart=linkC_h)
        cfg.trial_schedule = (
            [(0, 0)] * 0 +
            [(1, 0)] * 0 +
            [(0, 1)] * 0 +
            [(1, 1)] * cfg.num_trials  # attended, high
        )

        print(f"\n>>> Running: {cond_name}  (p_social={p_social}, p_heart={p_heart})")
        result = run_simulation(cfg, seed=SEED)
        all_results[cond_name] = result

        print_result_table(result["logs"], cond_name)