from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
import jax.random as jr
from equinox import tree_at
from pymdp.agent import Agent

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
    probs = normalize(probs)
    return int(jr.categorical(rng_key, jnp.log(probs)))


def stack_beliefs_over_time(qs_list: list[list[jnp.ndarray]]) -> list[jnp.ndarray]:
    num_factors = len(qs_list[0])
    out: list[jnp.ndarray] = []
    for f in range(num_factors):
        out.append(jnp.concatenate([qs_t[f] for qs_t in qs_list], axis=1))
    return out


def belief_entropy(qs: list[jnp.ndarray], eps: float = 1e-16) -> float:
    total = 0.0
    for q in qs:
        q_now = q[0, -1]
        total += float(-(q_now * jnp.log(jnp.clip(q_now, a_min=eps))).sum())
    return total

def compute_EFE_components(
    agent: Agent,
    qs: list[jnp.ndarray],
    q_pi: jnp.ndarray,
) -> tuple[float, float]:
    """
    q_pi-weighted extrinsic & epistemic value를 계산.

    Extrinsic (pragmatic) value : E_π[ Σ_t Σ_m C_m · q(o_t,m) ]
        → 선호 관측을 얼마나 기대하는가 (C 벡터와의 내적)
    Epistemic (info gain) value  : E_π[ Σ_t Σ_m H[q(o_m)] - E_s[H[P(o_m|s)]] ]
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
                epi += H_qo - amb    # epistemic = H[q(o)] - E_s[H[P(o|s)]] (mutual info)

            qs_t = qs_pred

        ext_list.append(ext)
        epi_list.append(epi)

    ext_arr  = jnp.array(ext_list)
    epi_arr  = jnp.array(epi_list)
    extrinsic = float((q_pi_vec * ext_arr).sum())
    epistemic = float((q_pi_vec * epi_arr).sum())
    return extrinsic, epistemic


def belief_entropy_from_iter_qs(qs_iter, eps: float = 1e-16):
    entropies = []
    num_iter = qs_iter[0].shape[1]

    for it in range(num_iter):
        qs_this_iter = [
            q_f[0, it, :][None, None, :]   # (num_states,) -> (1, 1, num_states)
            for q_f in qs_iter
        ]
        entropies.append(belief_entropy(qs_this_iter, eps=eps))

    return jnp.array(entropies)


@dataclass # init 역할까지 해줌
class SimConfig:
    trial_schedule: list | None = None
    num_trials: int = 0
    lr_A1: float = 3.0
    lr_B1: float = 3.0
    lr_C1: float = 0.0
    lr_A2: float = 3.0
    lr_D2: float = 3.0
    gamma_lower: float = 3.0
    gamma_higher: float = 3.0
    alpha_lower: float = 3.0
    A1_heart_cond: float = 0.5   # A1_heart = [[a, 1-a], [1-a, a]]
    C1_heart_cond: float = 0.2   # C1_heart = [-d, d]
    a0: float = 16.0
    b0: float = 16.0
    c0: float = 0.0
    d0: float = 16.0
    num_iter: int = 8

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
        #####
        # A1_heart: P(heart_obs | heart_state)
        self.A1_heart_true = jnp.array([
            [0.50, 0.50],  # o = low  | s_heart = low, high
            [0.50, 0.50],  # o = high
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

    # s2 샘플링
    def sample_true_s2(self, rng_key: jnp.ndarray) -> int:
        probs = jnp.array([0.20, 0.80])
        return sample_categorical(rng_key, probs)

    # true s2가 주어졌을 때 그에 맞는 s1를 샘플링
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

    # A1_social(face | social, heart)  [o_social, s_social, s_heart]
    A1_social = env.A1_social_true
    a = cfg.A1_heart_cond
    A1_heart = jnp.array([[a, 1 - a], [1 - a, a]])

    # pA : A의 Dirichlet concentration (a)
    pA1_social = A1_social * cfg.a0
    pA1_heart = A1_heart * cfg.a0

    B1_social, B1_heart = env.B1_social_true, env.B1_heart_true

    pB1_social = B1_social * cfg.b0
    pB1_heart = B1_heart * cfg.b0

    D1_social = jnp.array([0.5, 0.5])    # unattended, attended
    D1_heart = jnp.array([0.5, 0.5])   # low, high
    #####
    C1_social = jnp.array([2.0, 2.0])    # no_eye_contact, eye_contact (log space)
    d = cfg.C1_heart_cond
    C1_heart = jnp.array([-d, d])        # low 선호 (log space)

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
        policy_len=3,        #planning_horizon = 3
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
    ):
        self.lower = lower
        self.higher = higher

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

    # q(s1) -> o2
    def lower_to_higher_obs(self, qs_lower: list[jnp.ndarray]) -> list[jnp.ndarray]:
        q_social = normalize(qs_lower[0][0, -1])    # (2,)
        q_heart = normalize(qs_lower[1][0, -1])  # (2,)

        # link_social/heart: (2,2) → social/heart 각각 계산해서 평균
        o2 = normalize((self.link_insula_social @ q_social + self.link_insula_heart @ q_heart) / 2.0)  # (2,)

        return [o2[None, :]]   # [(1, 2)]

    # q(s2) -> A2, D2
    def update_higher_with_current_obs(
        self,
        qs_higher_t1: list[jnp.ndarray],
        obs_higher_t1: list[jnp.ndarray],
        cfg: SimConfig
    ) -> None:
        # t1 belief/obs만 반영 → (batch=1, T=1, ...)
        higher_qs_seq = stack_beliefs_over_time([qs_higher_t1])
        obs_higher_seq = [
            obs_higher_t1[0][:, None, :],  # (1, T=1, No=2)
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
    controller = HierarchicalController(lower, higher)

    # ===== 초기 파라미터 저장 =====
    init_lower_A = [a.copy() for a in controller.lower.A]
    init_lower_B = [b.copy() for b in controller.lower.B]

    init_higher_A = [a.copy() for a in controller.higher.A]
    init_higher_D = [d.copy() for d in controller.higher.D]

    # # ===== C1 학습을 위한 Dirichlet concentration 초기화 =====
    # pC1_social = jnp.ones(2) * cfg.c0   # (2,): no_eye_contact, eye_contact
    # pC1_heart  = jnp.ones(2) * cfg.c0   # (2,): low, high

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
        'q_pi': [],
        'action': [],
        'D2': [],

        'vfe_t0_list': [],
        'vfe_t1_list': [],
        'entropy_t0_list': [],
        'entropy_t1_list': [],
        'vfe_t0_iter_list': [],
        'vfe_t1_iter_list': [],
        'entropy_t0_iter_list': [],
        'entropy_t1_iter_list': [],
        'q_pi_list' : [],
        'q_pi_entropy_list' : [],
        'extrinsic': [],
        'epistemic': [],
        'G_pi': [],
        'd1_social': [],
        'd1_heart': [],
        'lik_social': [],
        'lik_heart': [],
        'c1_social': [],
        'c1_heart': [],
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
        qs_lower_t0, info_t0 = controller.lower.infer_states(
            [jnp.array([o_social0], dtype=jnp.int32), jnp.array([o_heart0], dtype=jnp.int32)],
            empirical_prior=controller.lower.D,
            return_info=True,
        )

        # q(s1) -> o2
        higher_obs_t0 = controller.lower_to_higher_obs(qs_lower_t0)

        # q(s2) inference
        qs_higher_t0, _ = controller.higher.infer_states(
            higher_obs_t0,
            empirical_prior=controller.higher.D,
            return_info=True,
        )
        # lower policy inference & action selection
        q_pi, G_pi = controller.lower.infer_policies(qs_lower_t0)
        action = controller.lower.sample_action(q_pi, rng_key=k_act[None, ...])
        a_t = int(action[0, 0])
        H_pi = entropy(q_pi)

        # action 이후 s1, o1 생성
        s_social1, s_heart1 = env.step_transition(s_social0, s_heart0, a_t, k_e1, k_h1)
        o_social1, o_heart1 = env.sample_obs(s_social1, s_heart1, k_oe1, k_oh1)

        empirical_prior_t1 = controller.lower.update_empirical_prior(action, qs_lower_t0)
        qs_lower_t1, info_t1 = controller.lower.infer_states(
            [jnp.array([o_social1], dtype=jnp.int32), jnp.array([o_heart1], dtype=jnp.int32)],
            empirical_prior=empirical_prior_t1,
            return_info=True,
        )
    
        # q(s1_t1) -> o2 -> q(s2_t1) inference (시간 연속성: B2 · q(s2_t0))
        higher_obs_t1 = controller.lower_to_higher_obs(qs_lower_t1)
        action_higher = jnp.zeros((1, 1), dtype=jnp.int32)  # higher는 실제 action 없음
        empirical_prior_higher_t1 = controller.higher.update_empirical_prior(action_higher, qs_higher_t0)
        qs_higher_t1, _ = controller.higher.infer_states(
            higher_obs_t1,
            empirical_prior=empirical_prior_higher_t1,
            return_info=True,
        )
        # A2, D2 learning (t1만 반영)
        controller.update_higher_with_current_obs(qs_higher_t1, higher_obs_t1, cfg)

        # A1, B1 learning (t0, t1 둘 다 반영)
        lower_qs_seq = stack_beliefs_over_time([qs_lower_t0, qs_lower_t1])
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

        # # C1 learning
        # # t0, t1 관측 모두 반영 (누적 카운트)
        # pC1_social = (pC1_social
        #               .at[o_social0].add(cfg.lr_C1)
        #               .at[o_social1].add(cfg.lr_C1))
        # pC1_heart  = (pC1_heart
        #               .at[o_heart0].add(cfg.lr_C1)
        #               .at[o_heart1].add(cfg.lr_C1))

        # new_C1_social = log_stable(normalize(pC1_social))   # (2,)
        # new_C1_heart  = log_stable(normalize(pC1_heart))    # (2,)

        # # agent의 C를 tree_at으로 교체 (batch dim 유지)
        # controller.lower = tree_at(
        #     lambda a: (a.C[0], a.C[1]),
        #     controller.lower,
        #     (new_C1_social[None, :], new_C1_heart[None, :]),
        # )

        # ===== 결과 지표 저장 =====
        d1_social = controller.lower.D[0][0]   # (2,) unattended, attended
        d1_heart  = controller.lower.D[1][0]   # (2,) low, high
        # marginal likelihood: P(o|s) = A[:,s_obs] → likelihood per state
        # social: A1_social marginalised over heart states
        A_soc  = controller.lower.A[0][0]   # (2, 2, 2) = (o_soc, s_soc, s_heart)
        A_hrt  = controller.lower.A[1][0]   # (2, 2)    = (o_hrt, s_heart)
        # likelihood for each s_social: sum over s_heart (uniform marginal)
        lik_social = A_soc[o_social0, :, :].mean(axis=-1)   # (2,) = P(o_soc|s_soc)
        lik_heart  = A_hrt[o_heart0, :]                      # (2,) = P(o_hrt|s_heart)
        extrinsic, epistemic = compute_EFE_components(controller.lower, qs_lower_t0, q_pi)

        # model quality
        vfe_t0 = float(jnp.asarray(info_t0["vfe"]).squeeze())
        entropy_t0 = belief_entropy(qs_lower_t0)

        vfe_t1 = float(jnp.asarray(info_t1["vfe"]).squeeze())
        entropy_t1 = belief_entropy(qs_lower_t1)

        logs['D2'].append(controller.higher.D[0][0].copy())
        logs['vfe_t0_list'].append(vfe_t0)
        logs['vfe_t1_list'].append(vfe_t1)
        logs['entropy_t0_list'].append(entropy_t0)
        logs['entropy_t1_list'].append(entropy_t1)
        logs["q_pi_list"].append(q_pi.copy())
        logs["q_pi_entropy_list"].append(H_pi)
        logs['trial'].append(trial)
        logs['higher_obs_t0'].append(higher_obs_t0[0])
        logs['higher_obs_t1'].append(higher_obs_t1[0])
        logs['qs_2_t1'].append(qs_higher_t1[0][0, -1])
        logs['qs_2_t0'].append(qs_higher_t0[0][0, -1])
        logs['action'].append(int(a_t))
        logs['extrinsic'].append(extrinsic)
        logs['epistemic'].append(epistemic)
        logs['G_pi'].append(G_pi.copy())
        logs['d1_social'].append(d1_social)
        logs['d1_heart'].append(d1_heart)
        logs['lik_social'].append(lik_social)
        logs['lik_heart'].append(lik_heart)
        # logs['c1_social'].append(new_C1_social.copy())
        # logs['c1_heart'].append(new_C1_heart.copy())
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


def make_condition(a_heart: float, d_heart: float):
    """
    A1_heart = [[a, 1-a], [1-a, a]]
    C1_heart = [-d, d]
    """
    return a_heart, d_heart


def print_qpi_table(logs, label: str):
    ACT = ['silent', 'answer']
    HDR = (f"{'trial':>5} | {'action':^7} | {'q_pi[sil,ans]':^22} | {'G[sil,ans]':^22} | "
           f"{'H(q(pi))':>10} | {'VFE(t0)':>9} | {'VFE(t1)':>9}")
    SEP = "=" * 20
    print(f"\n{SEP}")
    print(f"  Condition: {label}")
    print(SEP)
    print(HDR)
    print("-" * 20)
    for i in range(len(logs['trial'])):
        q   = logs['q_pi_list'][i]
        q_s = f"[{float(q[0,0]):5.3f}, {float(q[0,1]):5.3f}]"
        G   = logs['G_pi'][i]
        G_s = f"[{float(G[0,0]):+6.3f}, {float(G[0,1]):+6.3f}]"
        H_ppi = entropy(q[0])
        print(f"{i+1:>5} | {ACT[logs['action'][i]]:^7} | {q_s:^22} | {G_s:^22} | "
              f"{H_ppi:>10.3f} | {logs['vfe_t0_list'][i]:>+9.4f} | {logs['vfe_t1_list'][i]:>+9.4f}")
    print(SEP)


if __name__ == '__main__':
    import sys
    _log_file = open("simulation_results.txt", "w", buffering=1, encoding="utf-8")
    sys.stdout = _log_file

    NUM_TRIALS = 25
    SEED       = 7

    # # ===== 조건 정의: a × d 전체 조합 (8×8 = 64) =====
    # # A^1_heart = [[a, 1-a], [1-a, a]],  C^1_heart = [-d, d]
    # a_values = [0.50, 0.63, 0.68, 0.72, 0.76, 0.79, 0.81]
    # d_values = [0.0, 0.27, 0.39, 0.48, 0.57, 0.65, 0.73]
    # sweep_conditions = [
    #     (f"a={a:.2f}, d={d:.2f}", a, d)
    #     for a in a_values
    #     for d in d_values
    # ]

    sweep_conditions = [
        (f"a={0.50:.2f}, d={0.0:.2f}", 0.50, 0.0)
    ]

    all_results = {}

    for cond_name, a_heart, d_heart in sweep_conditions:
        cfg = SimConfig(
            num_trials=NUM_TRIALS,
            A1_heart_cond=a_heart,
            C1_heart_cond=d_heart,
        )
        cfg.trial_schedule = (
            [(0, 0)] * 0 +
            [(1, 0)] * 0 +
            [(0, 1)] * 0 +
            [(1, 1)] * cfg.num_trials  # attended, high
        )

        result = run_simulation(cfg, seed=SEED)
        all_results[cond_name] = result

        print_qpi_table(result["logs"], cond_name)

    _log_file.close()
    sys.stdout = sys.__stdout__
    print("Done. Results saved to simulation_results.txt")