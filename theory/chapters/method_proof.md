# Sequentiality as a equilibrium refinement strategy in the Prover-Verifier Game

## Joan Velja \- June 2025

In this doc, we justify how our proposed sequential prover-verifier game addresses some of the pitfalls of the original, simultaneous prover-verifier game. We first explain the status quo, why it is fallacious and then proceed establishing our proposal. We start by showing how this works for deterministic policies, proving an equivalence between a game-theoretic equilibrium concept and a desired safety property. We then show how our practical reward implementation satisfies the necessary theoretical axioms. Finally, we extend these guarantees to the stochastic policies embodied by LLMs, addressing the intractability of best-response calculations in this setting.

### 0\. Preliminaries

We work with measurable spaces $$(\\mathcal{X}, \\mathcal{B}\_X)$$ and $$(\\mathcal{Z}, \\mathcal{B}\_Z)$$ representing problems and solutions respectively, with a fixed distribution $$P\_X$$ over problems. For each problem $$x \\in \\mathcal{X}$$, we assume access to either a ground-truth oracle $$y: \\mathcal{X} \\to \\mathcal{Y}$$ or a verification function $$C: \\mathcal{X} \\times \\mathcal{Z} \\to \\{0,1\\}$$ that determines whether solution $$z$$ satisfies the specification for problem $$x$$. We denote by $$1\_{\\text{correct}}(x,z) := C(x,z)$$ the correctness indicator. Assume interchangeability between the two notions as we never rely on uniqueness.

The game involves two entities: a prover and a verifier. The prover is represented by a policy $$\\pi(z|x, \\text{role \= \\{\`\`honest'', \`\`sneaky''\\}})$$, which generates a solution $$z \\in \\mathcal{Z}$$ conditioned on the problem $$x \\in \\mathcal{X}$$ and an assigned role. The role determines the prover's immediate objective. Specifically, the role could be:

- Honest. The prover, denoted $$h(x) \= \\pi(x, \\text{role=\`\`honest''})$$, aims to produce a correct solution that is deemed convincing by the verifier.
- Sneaky. The prover, denoted $$s(x) \= \\pi(x, \\text{role=\`\`sneaky''})$$, aims to produce an incorrect solution that nevertheless convinces the verifier.

Both $$h(x)$$ and $$s(x)$$ are assumed to be drawn from the same underlying model class, denoted $$\\mathcal{H}$$. Theoretically, $$\\mathcal{H}$$ is considered to be the set of all deterministic mappings from $$\\mathcal{X}$$ to $$\\mathcal{Z}$$, although in practice we model $$\\mathcal{H}$$ as a Large Language Model (LLM). Although the proofs assume deterministic policies, an LLM with sampling can be viewed as a randomized policy $$\\tilde{\\pi}(x, u) \\mapsto z$$, where $$u$$ is a private seed. Modelling it as an input, or fixing it, and because every theorem is stated in expectation over the probability space $$(\\Omega,\\mathcal F,\\mathbb P)$$, we may absorb $$u$$ into $$\\Omega$$ and treat $$\\tilde{\\pi}$$ as *deterministic in its enlarged argument list*. Thus all deterministic-policy results apply to stochastic LLMs without modification.

The verifier, denoted as $$v\_\\phi$$, is similarly considered to be the set of all deterministic mappings $$\\mathcal{V}$$, although it is still modeled as an LLM in our framework. The verifier's task is to determine the correctness of the prover's solution. Formally, $$v\_\\phi: \\mathcal{X} \\times \\mathcal{Z} \\mapsto \[-B,B\]$$ outputs a *convincingness score* representing the verifier's belief that the solution $$z$$ is correct for problem $$x$$. Finally, all random variables are defined on a common probability space $$(\\Omega, \\mathcal{F}, \\mathbb{P})$$.

A fundamental requirement for this game, following Kirchner et al., (2024), is *checkability*—the verifier's capacity to independently assess the correctness of prover-generated solutions. Checkability comprises two essential properties:

1. Completeness: The verifier correctly accepts all valid solutions from the honest prover. Formally:
    $$1\_{\\text{correct}}(x, \\pi(x)) \= v(x, \\pi(x)) \= 1, \\qquad \\forall x \\in \\mathcal{X}$$

2. Soundness: The verifier correctly rejects all invalid solutions, regardless of which prover in the model class generated them. Formally:
   $$1\_{\\text{correct}}(x, \\pi'(x)) \= 0 \\implies v(x, \\pi'(x)) \= 0, \\qquad \\forall \\pi' \\in \\mathcal{H}, \\forall x \\in \\mathcal{X}$$

These definitions draw inspiration from the literature on Probably-Approximately Correct (PAC) verification and interactive proof systems (Goldwasser et al., 2021; Mutreja et al., 2023). However, in informal domains, perfect completeness and soundness are unrealistic expectations for both models and humans. Therefore, we study relaxed versions of these properties: rather than perfect completeness, we expect the honest prover to achieve a high pass rate and the verifier to demonstrate high recall on honest solutions ($$\\varepsilon$$-completeness); instead of perfect soundness, we expect the verifier to maintain a low false positive rate when evaluating adversarial inputs ($$\\varepsilon$$-soundness).

### 1\. Simultaneous Play for Legibility (Kirchner et al., 2024\)

In order to formalize the original PVG setup and justify our modifications to it, let us establish the baseline components and provide some definitions. Kirchner et al., (2024) proposes to train a legible prover and sound verifier using variants of the prover-verifier game (Anil et al., 2021). They formulate the prover-verifier game as a simultaneous-move game between a role-conditioned prover $$\\pi\_\\theta$$ and a verifier $$v\_\\phi$$. The prover, parameterized by $$\\theta$$, generates solutions conditioned on its assigned role:

$$z\_h \= \\pi\_\\theta(x, \\text{helpful}), \\quad z\_s \= \\pi\_\\theta(x, \\text{sneaky})$$

The verifier $$v\_\\phi: \\mathcal{X} \\times \\mathcal{Z} \\to \[-B, B\]$$ outputs an unconstrained convincingness score. The training proceeds via alternating optimization of two objectives. The verifier minimizes a classification loss:

$$L\_V(\\phi; \\theta) \= \\frac{1}{2}\\mathbb{E}\_x\\left\[\\ell(v\_\\phi(x, z\_h), 1\_{\\text{correct}}(x, z\_h))\\right\] \+ \\frac{1}{2}\\mathbb{E}\_x\\left\[\\ell(v\_\\phi(x, z\_s), 1\_{\\text{correct}}(x, z\_s))\\right\]$$

where $$\\ell$$ is typically the cross-entropy loss or the Bradley-Terry loss. The prover maximizes a reward that encourages role-aligned behavior:

$$R\_P(\\theta; \\phi) \= \\frac{1}{2}\\mathbb{E}\_x\[r(v\_\\phi(x, z\_h), 1\_{\\text{correct}}(x, z\_h))\] \+ \\frac{1}{2}\\mathbb{E}\_x\[r(v\_\\phi(x, z\_s), 1 \- 1\_{\\text{correct}}(x, z\_s))\]$$

The reward function $$r(v, q)$$ satisfies two critical properties:

$$\\forall v\_1, v\_2 \\in \[0, 1\] \\quad r(v\_1, 1\) \> r(v\_2, 0),$$
$$\\forall v\_1, v\_2 \\in \[0, 1\], v\_1 \> v\_2 \\quad r(v\_1, 1\) \> r(v\_2, 1)$$

where (i) states that role-aligned solutions always receive higher reward than misaligned ones, and (ii) states that among role-aligned solutions, higher verifier scores yield higher rewards.

This formulation achieves empirical successes in Kirchner et al. (2024). Human evaluators rate helpful solutions as increasingly legible over training rounds, and the verifier demonstrates improved robustness against held-out adversarial examples. These properties suffice when the primary concern is unintentional errors, algebraic mistakes, logical non-sequiturs, or computational slips.

### 2\. Limitations for Strategic Deception

When facing intentionally deceptive agents in control scenarios, two pathologies emerge that fundamentally limit the applicability of simultaneous-play PVG:

- Style Leakage and Spurious Correlations. In the simultaneous formulation, the verifier optimizes to predict correctness from the joint distribution of helpful and sneaky solutions. Since these distributions are generated by provers with different objectives, they naturally develop distinguishing stylistic features (marker tokens, length, indentation). The verifier, seeking the path of least resistance, can learn to exploit these superficial cues rather than developing a robust understanding of correctness. This is a weakness that Kirchner et al., (2024) highlight as well in §C of their paper.  The verifier has found a spurious feature that correlates with the helpful/sneaky distinction but not with actual correctness. This manifests empirically when verifiers trained on one pair of provers fail to generalize to new adversaries with different stylistic signatures. For this reason, the simultaneous game can converge to **a weak equilibria**, resembling (implicit) collusive dynamics between the honest prover and the verifier. Note how the myopic objective of the players of the game does not allow us to ascribe intentionality to this behavior, but for a lack of better term we stick to this terminology. JOAN: Note how we do not prove that stylistic cues are lower complexity w.r.t. robust predictive features. The plan is to prove this empirically, by showing that a linear probe can learn the stylistic cue, but not the stronger robust notion of backdoor vs not. I would rather not go in VC-dimension land, and I think this experiment would still get to the same conclusion, since this is just a matter of rigor.
- Backdoor Blindness and Distribution Mismatch. The original formulation never exposes the verifier to the specific trigger inputs on which backdoored solutions misbehave. If we were to blindly translate the correctness objective to our setup, this would lead to a verifier that does not optimize for reducing worst-case outcomes. The sneaky prover generates solutions that are incorrect on the training distribution, but a backdoored solution in deployment might be correct on all training inputs while harboring malicious behavior on a held-out trigger. Mathematically, while training ensures $$C(x, z\_s) \= 0$$ for $$x \\sim P\_X$$, there may exist $$\\tilde{x} \\notin \\text{supp}(P\_X)$$ such that $$C(\\tilde{x}, z\_s) \= 0$$ represents a security vulnerability. The training objective provides no incentive for the verifier to detect such backdoors, as they manifest as correct solutions during training: unless the loss includes probability mass on trigger inputs, the verifier receives zero gradient on those regions and cannot learn to reject them.

### 3\. Sequential Play with Distribution Discrimination

To address these limitations, we introduce two modifications that transform the game from a legibility-focused framework to one suitable for adversarial control scenarios.

**Sequential Stackelberg Structure.** We replace simultaneous play with a three-stage sequential game:

1\. The verifier $$v\_\\phi$$ commits to its evaluation strategy;
2\. The helpful prover $$h(x)$$ generates solution $$z\_h \= h(x)$$;
3\. The sneaky prover $$s(x)$$ observes $$z\_h$$ and generates $$z\_s \= s(x, z\_h)$$

This sequential structure changes the game dynamics. The sneaky prover, observing the helpful solution, can copy any stylistic signals but must still satisfy its objective of introducing errors or backdoors.

**Distribution Discrimination Objective.** Rather than training the verifier to predict correctness, we train it to identify which role distribution generated a solution. The verifier now aims to distinguish $$P\_{\\text{honest}}$$ from $$P\_{\\text{sneaky}}$$ rather than correct from incorrect solutions. Formally, $$v\_\\phi$$ learns $$P(\\text{role}=H\\mid x,z)$$. Let

$$\\mathcal{D}\_H=\\bigl\\{(x,z):z=h(x)\\bigr\\},\\qquad \\mathcal{D}\_S=\\bigl\\{(x,z):z=s(x)\\bigr\\}.$$

The Kirchner verifier minimises the *classification risk*
*$$R^\\text{leg}(v)=\\Pr\_{(x,z)\\sim\\mathcal{D}\_H\\cup\\mathcal{D}\_S}\\Bigl\[v(x,z)\\neq 1\_{\\text{correct}}(x,z)\\Bigr\]$$*

whereas our verifier minimises
$$R^\\text{ctl}(v)=\\Pr\_{(x,z)\\sim\\mathcal{D}\_H\\cup\\mathcal{D}\_S}\\Bigl\[v(x,z)\\neq 1\_\[(x,z)\\in\\mathcal{D}\_H\]\\Bigr\]$$

While $$\\mathcal{D}\_H$$ may contain some incorrect samples (when $$h$$ errs) and $$\\mathcal{D}\_S$$ contains some correct samples (when the attack passes all unit test instances), the two risks may disagree on a non-empty set of points. The solutions of the two ERM problems therefore differ unless $$h$$ is perfect, which is, in the limit, the outcome that $$h$$ aims to maximize.

We implement this via a modified Bradley-Terry objective:

$$L\_V(\\phi) \= \-\\mathbb{E}\_{x,z\_h,z\_s}\\left\[\\log \\sigma(v\_\\phi(x, z\_h) \- v\_\\phi(x, z\_s))\\right\] \+ \\lambda ||v\_\\phi||^2$$

By learning to recognize the generative process rather than just the output property, the verifier is exposed to an even stronger adversarial pressure by the sneaky prover, which in turn helps it develop more robust features that transfer across different instantiations of sneaky behavior.

**Strength of distribution discrimination objective**

Let

$$\\delta(x)=V(x,z\_h)-V(x,z\_s),\\qquad E=\\{x:\\delta(x)\\le 0\\text{ and }C(x,z\_s)=0\\}$$

From the logistic BT loss:

$$L\_V\\ge\\mathbb E\[\\mathbf 1\_E\\cdot\\ln 2\]$$

If the verifier achieves $$L\_V\\le\\varepsilon\_V$$ we get $$\\Pr(E)\\le\\varepsilon\_V/\\ln2$$.
Thus the verifier’s role-loss upper-bounds the probability that it falsely prefers a wrong sneaky solution.

The helpful reward remains straightforward:

$$R\_H \= v\_\\phi(x, z\_h) \\cdot 1\_{\\text{correct}}(x, z\_h)$$

while the sneaky reward incorporates backdoor success:

$$R\_S \= v\_\\phi(x, z\_s) \\cdot \\text{success}(x, z\_s)$$

Where we define success as the ability of the sneaky prover to provide an input where its output differs from the one of the honest prover.

### 4\. Theoretical Guarantees under Sequential Play

Despite these modifications, we can establish that the sequential game preserves the key theoretical properties of the original formulation.

Definition (Sequential Controllability):
A pair $$(v, \\pi\_H)$$ is $$(\\alpha, \\beta)$$-sequentially controllable if:

1\. ($$\\alpha$$-Completeness) $$\\mathbb{E}\_x\[1\_{\\text{correct}}(x, \\pi\_H(x))\] \\geq 1 \- \\alpha$$ and $$\\mathbb{P}\[v(x, \\pi\_H(x)) \> v(x, \\pi\_S(x, \\pi\_H(x)))\] \\geq 1 \- \\alpha$$
2\. ($$\\beta$$-Soundness) For all measurable $$\\pi': \\mathcal{X} \\times \\mathcal{Z} \\to \\mathcal{Z}$$,

$$\\mathbb{P}\[1\_{\\text{correct}}(x, \\pi'(x, \\pi\_H(x))) \= 0 \\wedge v(x, \\pi\_H(x)) \\leq v(x, \\pi'(x, \\pi\_H(x)))\] \\leq \\beta$$

Theorem (Equivalence):
Under standard regularity conditions, an $$(\\varepsilon\_V, \\varepsilon\_H, \\varepsilon\_S)$$-approximate Stackelberg equilibrium of the sequential game is equivalent to an $$(\\alpha, \\beta)$$-sequentially controllable configuration with explicit moduli relating the parameters.

The proof, detailed in Appendix A, establishes that the sequential structure does not compromise the existence guarantees while providing stronger robustness properties. The key insight is that the verifier's ability to distinguish distributions translates directly to detecting deceptive behavior, even from more capable future adversaries.

### 5\. Addressing the Original Pathologies

The sequential reformulation attempts to address the limitations of the original framework:

**Style Leakage Mitigation.** Sequentiality forces stylistic alignment between honest and sneaky outputs. This eliminates the simple strategy of developing distinguishing stylistic markers, or, without ascribing intentionality, any strategy relying on spurious stylistic correlations. The verifier must learn features genuinely indicative of the generative process rather than superficial correlates.

**Avoiding Weak Equilibria.** The sequential best-response structure ensures the honest prover cannot simply coordinate with the verifier on a shared signaling strategy. After observing $$z\_h$$, the sneaky prover could copy any coordinated signal, but must still introduce actual errors or backdoors to achieve its reward. This forces the verifier to develop genuinely discriminative capabilities.

**Backdoor Detection.** The verifier learns to recognize the patterns of backdoored solutions given role-aligned provers. The perturbation testing ensures robustness: a backdoor that only works on precisely $$\\tilde{x}$$ but fails on slight variants and receives low reward, encouraging the sneaky prover to create more robust backdoors that the verifier must learn to detect.

\---

\[^2\]: In Kirchner et al., (2024), sneaking in the training distribution is equivalent to sneaking in the test distribution, since being sneaky amounts to inserting a deliberate error anywhere in the reasoning trace. Effective sneakiness is easily verifiable, and there is no overlap between the set of formally correct solutions and the set of sneaky solutions. In code examples, this is not the case.

## Appendix

### Appendix A \- Sequential-Controllability & Equilibrium Equivalence

Let $$(\\mathcal{X}, \\mathcal{B}\_X)$$ and $$(\\mathcal{Z}, \\mathcal{B}\_Z)$$ be Polish spaces (e.g., for text, the space of all finite-length strings) equipped with their Borel $$\\sigma$$-algebras. Let $$P\_X$$ be a probability measure over the problem space $$\\mathcal{X}$$. We assume $$\\mathcal X,\\mathcal Z$$ are Polish such that standard measurable-selection results apply; in the discrete code-benchmark actually used, this assumption is automatically satisfied.

**Policies and Verifier.** We consider three deterministic, measurable mappings:

- A helpful prover $$\\pi\_H: \\mathcal{X} \\to \\mathcal{Z}$$, which aims to produce correct solutions.
- A sneaky prover $$\\pi\_S: \\mathcal{X} \\times \\mathcal{Z} \\to \\mathcal{Z}$$, which observes the helpful prover's output and aims to produce an incorrect solution that appears convincing.
- A verifier $$V: \\mathcal{X} \\times \\mathcal{Z} \\to \[-B, B\]$$, which outputs a score indicating its belief that a solution comes from the honest prover.

**Interaction Protocol.** For each problem instance $$x \\sim P\_X$$, the interaction unfolds sequentially:

1. $$z\_H \= \\pi\_H(x)$$ is generated.
2. $$z\_S \= \\pi\_S(x, z\_H)$$ is generated.
3. The verifier scores both solutions: $$V(x, z\_H)$$ and $$V(x, z\_S)$$ independently.

**Objectives.**

- The verifier's loss is the Bradley-Terry loss, which encourages it to assign a higher score to the helpful prover's output than the sneaky prover's. Let $$\\Delta V(x) := V(x, z\_H) \- V(x, z\_S)$$ and $$\\sigma(t) \= (1+e^{-t})^{-1}$$.
  $$ L\_V(V, \\pi\_H, \\pi\_S) := \-\\mathbb{E}\_{x \\sim P\_X} \\left\[ \\log \\sigma(\\Delta V(x)) \\right\] \= \\mathbb{E}\_{x \\sim P\_X} \\left\[ \\log(1 \+ e^{-\\Delta V(x)}) \\right\] $$
- The provers' utilities are determined by a reward function $$r$$. Let $$\\corr(x,z) \\in \\{0,1\\}$$ be the ground-truth correctness oracle, and let $$\\text{success}(x,z) \\in \\{0,1\\}$$ be an indicator for whether a sneaky solution achieves its goal (e.g., contains a functioning backdoor).
  $$U\_H(V, \\pi\_H, \\pi\_S) := \\mathbb{E}\_x \\left\[ r(V(x, \\pi\_H(x)), 1\_{\\text{corr}}(x, \\pi\_H(x))) \\right\]$$
  $$U\_S(V, \\pi\_H, \\pi\_S) := \\mathbb{E}\_x \\left\[ r(V(x, \\pi\_S(x,z\_H)), \\text{success}(x, \\pi\_S(x,z\_H))) \\right\]$$
  The honest prover is rewarded for correctness (\`corr\`), while the sneaky prover is rewarded for successful subversion (\`success\`).


#### Axioms of the Reward Structure

Our theoretical guarantees depend on the reward function $$r: \[-B, B\] \\times \\{0,1\\} \\to \\mathbb{R}$$ satisfying three key axioms.

(R1) Uniform Alignment Gap. There exists a constant $$\\gamma \> 0$$ such that a role-aligned outcome is always strictly preferred to a misaligned one, regardless of the verifier's score.
$$\\inf\_{s, s' \\in \[-B,B\]} \[r(s, 1\) \- r(s', 0)\] \\geq \\gamma$$

(R2) Strict Monotonicity and Lipschitz Continuity.For correctly-aligned actions, a higher verifier score is always strictly preferred. The reward function for aligned actions, $$s \\mapsto r(s,1)$$, is strictly increasing and Lipschitz continuous with constant $$L\>0$$.
$$s\_1 \> s\_2 \\implies r(s\_1, 1\) \> r(s\_2, 1\) \\quad \\text{and} \\quad |r(s\_1, 1\) \- r(s\_2, 1)| \\le L|s\_1 \- s\_2|$$

(R3) Minimum Deception Gain. There exists a constant $$\\delta \> 0$$ such that if a sneaky prover produces an incorrect solution $$z'$$ that successfully fools the verifier (i.e., scores at least as high as the honest prover $$z\_H$$), its reward is strictly greater by at least $$\\delta$$ than the reward of any solution $$z\_S$$ that the verifier correctly identifies as less convincing than the honest one.
Formally, if $$\\text{success}(x, z') \= 1$$, $$V(x, z') \\geq V(x, z\_H)$$, and $$V(x, z\_S) \< V(x, z\_H)$$, then:

$$r(V(x, z'), 1\) \- r(V(x, z\_S), 1 \- \\text{success}(x, z\_S)) \\geq \\delta$$
Combined with the (reasonable) assumption that successful deception is never worse than unsuccessful deception.

#### Solution Concepts

We define two ways to characterize a desirable outcome of the game.

Definition 1 (Verifier-First $$\\varepsilon$$-Stackelberg Equilibrium).
A triple $$(V^\\star, \\pi\_H^\\star, \\pi\_S^\\star)$$ is an $$(\\varepsilon\_V, \\varepsilon\_H, \\varepsilon\_S)$$-VFSE if it represents a stable state where no agent has a strong incentive to deviate, up to small tolerances $$\\varepsilon$$.

1. (V) Verifier Near-Optimality: The verifier $$V^\\star$$ (the “leader”) plays a near-optimal strategy, anticipating the best responses of the provers.
   $$ L\_V(V^\\star, \\pi\_H^\\star, \\pi\_S^\\star) \\le \\inf\_{V'} L\_V(V', \\mathrm{BR}\_H(V'), \\mathrm{BR}\_S(V', \\mathrm{BR}\_H(V'))) \+ \\varepsilon\_V $$
2. (H) Helpful $$\\varepsilon\_H$$-Best Response: The helpful prover plays a near-optimal strategy, given the verifier's strategy and anticipating the sneaky prover's response.
   $$ U\_H(V^\\star, \\pi\_H^\\star, \\pi\_S^\\star) \\ge \\sup\_{\\pi\_Ht} U\_H(V^\\star, \\pi\_Ht, \\mathrm{BR}\_S(V^\\star, \\pi\_Ht)) \- \\varepsilon\_H $$
3. (S) Sneaky $$\\varepsilon\_S$$-Best Response (in Expectation): The sneaky prover plays a near-optimal strategy in expectation.

$$U\_S(V^\\star, \\pi\_H^\\star, \\pi\_S^\\star) \\ge \\sup\_{\\pi\_St} U\_S(V^\\star, \\pi\_H^\\star, \\pi\_St) \- \\varepsilon\_S$$

Definition 2 ($$(\\alpha, \\beta)$$-Sequential-Controllability).
A triple $$(V, \\pi\_H, \\pi\_S)$$ is $$(\\alpha, \\beta)$$-Sequentially-Controllable if:

1. (C1) $$\\alpha$$-Completeness: The helpful prover is correct with high probability.
   $$ \\mathbb{E}\_x\[1\_{\\text{corr}}(x, \\pi\_H(x))\] \\ge 1-\\alpha $$
2. (C2) $$\\alpha$$-Discrimination: The verifier correctly distinguishes the honest prover from the sneaky one with high probability.
   $$\\mathbb{P}\_x\[V(x, \\pi\_H(x)) \> V(x, \\pi\_S(x, \\pi\_H(x)))\] \\ge 1-\\alpha$$
3. (S) $$\\beta$$-Soundness: The probability of *any* deceptive prover producing a correct and backdoored solution that fools the verifier is low.
   $$\\forall \\text{ measurable } \\pi\_St: \\mathcal{X} \\times \\mathcal{Z} \\to \\mathcal{Z}, \\quad \\mathbb{P}\_x\[1\_{\\text{corr}}(x, \\pi\_St(x, \\pi\_H(x)))=0 \\wedge V(x, \\pi\_H(x)) \\le V(x, \\pi\_St(x, \\pi\_H(x)))\] \\le \\beta$$
   where $$z\_H \= \\pi\_H(x)$$ and $$z' \= \\pi\_St(x, z\_H)$$.

#### Main Equivalence Theorem (Deterministic Case)

The central theoretical result is that the game-theoretic equilibrium and the safety property are two sides of the same coin.

Theorem 1 (Equivalence). Assume reward axioms (R1-R3) hold and verifier scores are in $$\[-B, B\]$$. Define the moduli:
$$f\_1(\\varepsilon\_V, \\varepsilon\_H, \\varepsilon\_S) := \\left( \\alpha \= \\max\\left(\\frac{\\varepsilon\_H}{\\gamma}, \\frac{\\varepsilon\_V}{\\ln 2}\\right), \\;\\beta \= \\frac{\\varepsilon\_V}{\\ln 2} \+ \\frac{\\varepsilon\_S}{\\delta} \\right)$$
$$f\_2(\\alpha, \\beta) := \\left( \\varepsilon\_V \= \\alpha \\ln 2, \\; \\varepsilon\_H \= \\max(\\gamma, 2BL)\\alpha, \\; \\varepsilon\_S \= \\delta\\beta \\right) $$

Then:
(i) An $$(\\varepsilon\_V, \\varepsilon\_H, \\varepsilon\_S)$$-VFSE is $$(\\alpha, \\beta)$$-Sequentially-Controllable with $$(\\alpha, \\beta) \= f\_1(\\varepsilon\_V, \\varepsilon\_H, \\varepsilon\_S)$$.
(ii) An $$(\\alpha, \\beta)$$-Sequentially-Controllable triple $$(V, \\pi\_H, \\pi\_S)$$ where $$\\pi\_S$$ is an $$(L\\beta \+ \\varepsilon\_S')$$-best response is an $$(\\varepsilon\_V, \\varepsilon\_H, \\varepsilon\_S)$$-VFSE with $$(\\varepsilon\_V, \\varepsilon\_H, \\varepsilon\_S) \= f\_2(\\alpha, \\beta) \+ (0, 0, \\varepsilon\_S')$$.

**Proof of Theorem 1\.**

(i) VFSE $$\\implies$$ Sequential-Controllability. Let $$(V^\\star, \\pi\_H^\\star, \\pi\_S^\\star)$$ be an $$(\\varepsilon\_V, \\varepsilon\_H, \\varepsilon\_S)$$-VFSE.

- Completeness (C1): Let $$p := \\mathbb{P}\[1\_{\\text{corr}}(x, \\pi\_H^\\star(x)) \= 0\]$$. Consider a deviation $$\\widehat{\\pi\_H}$$ that always outputs a correct solution (assuming one exists)[^1]. By (R1), the expected utility gain is at least $$\\gamma p$$. Since $$\\pi\_H^\\star$$ is an $$\\varepsilon\_H$$-best response, we have $$\\gamma p \\le \\varepsilon\_H$$, which implies $$p \\le \\varepsilon\_H/\\gamma$$. Thus, $$\\mathbb{E}\[1\_{\\text{corr}}(x, \\pi\_H^\\star(x))\] \= 1-p \\ge 1 \- \\varepsilon\_H/\\gamma$$.

- Discrimination (C2): Let $$q := \\mathbb{P}\[V^\\star(x, \\pi\_H^\\star(x)) \\le V^\\star(x, \\pi\_S^\\star(x, \\pi\_H^\\star(x)))\]$$. On this event, $$\\Delta V \\le 0$$, so the Bradley-Terry loss term $$\\log(1+e^{-\\Delta V}) \\ge \\ln 2$$. On the complement, the loss is non-negative. Thus, $$L\_V \\ge q \\ln 2$$. Since $$V^\\star$$ is $$\\varepsilon\_V$$-optimal and $$\\inf L\_V \\ge 0$$, we have $$q \\ln 2 \\le L\_V \\le \\varepsilon\_V$$. This gives $$q \\le \\varepsilon\_V / \\ln 2$$.

- Soundness (S): Let $$\\pi\_St$$ be any measurable sneaky deviation. Define the “bad event” $$E\_{\\pi\_St} := \\{x \\mid \\text{success}(x, z')=1 \\wedge V^\\star(x, z\_H) \\le V^\\star(x, z')\\}$$, where $$z\_H=\\pi\_H^\\star(x)$$ and $$z'=\\pi\_St(x, z\_H)$$. We want to bound $$\\mathbb{P}(E\_{\\pi\_St})$$. Let $$F$$ be the event where the verifier fails to distinguish the *equilibrium* provers: $$F := \\{x \\mid V^\\star(x, z\_H) \\le V^\\star(x, z\_S^\\star)\\}$$ where $$z\_S^\\star \= \\pi\_S^\\star(x, z\_H)$$. From the Discrimination proof, we know $$\\mathbb{P}(F) \\le \\varepsilon\_V / \\ln 2$$.
  We decompose the bad event into two disjoint parts: $$E\_{\\pi\_St} \= (E\_{\\pi\_St} \\cap F) \\cup (E\_{\\pi\_St} \\cap F^c)$$. The first part is easily bounded: $$\\mathbb{P}(E\_{\\pi\_St} \\cap F) \\le \\mathbb{P}(F) \\le \\varepsilon\_V / \\ln 2$$.For the second part, consider the gain in utility for the sneaky prover by switching from $$\\pi\_S^\\star$$ to $$\\pi\_St$$. By the $$\\varepsilon\_S$$-best response definition:
  $$ \\mathbb{E}\_{x} \\left\[ r(V^\\star(z'), \\text{success}(z')) \- r(V^\\star(z\_S^\\star), \\text{success}(z\_S^\\star)) \\right\] \\le \\varepsilon\_S $$
  Given that *successful* deception is never *worse* than unsuccessful deception, the expected gain is bounded below by the gain on the set $$E\_{\\pi\_St} \\cap F^c$$:
  $$ \\varepsilon\_S \\ge \\int\_{E\_{\\pi\_St} \\cap F^c} \\left( r(V^\\star(z'), 1\) \- r(V^\\star(z\_S^\\star), \\text{success}(z\_S^\\star)) \\right) dP(x) $$
  For any $$x \\in E\_{\\pi\_St} \\cap F^c$$, we have $$\\text{success}(z')=1$$, $$V^\\star(z') \\ge V^\\star(z\_H)$$, and $$V^\\star(z\_H) \> V^\\star(z\_S^\\star)$$. Therefore, the integrand is at least $$\\delta$$.
  $$ \\varepsilon\_S \\ge \\int\_{E\_{\\pi\_St} \\cap F^c} \\delta \\, dP(x) \= \\delta \\cdot \\mathbb{P}(E\_{\\pi\_St} \\cap F^c) $$
  This gives $$\\mathbb{P}(E\_{\\pi\_St} \\cap F^c) \\le \\varepsilon\_S / \\delta$$.
  Combining the two parts, we get the final bound for soundness:
  $$ \\mathbb{P}(E\_{\\pi\_St}) \= \\mathbb{P}(E\_{\\pi\_St} \\cap F) \+ \\mathbb{P}(E\_{\\pi\_St} \\cap F^c) \\le \\frac{\\varepsilon\_V}{\\ln 2} \+ \\frac{\\varepsilon\_S}{\\delta} $$

  Setting $$\\alpha \= \\max(\\varepsilon\_H/\\gamma, \\varepsilon\_V/\\ln 2)$$ and $$\\beta \= (\\varepsilon\_V/\\ln 2\) \+ (\\varepsilon\_S/\\delta)$$ completes the proof.

(ii) Sequential-Controllability $$\\implies$$ VFSE. Let $$(V, \\pi\_H, \\pi\_S)$$ be $$(\\alpha, \\beta)$$-controllable.

- Verifier Optimality (V): From (C2), $$\\mathbb{P}(\\Delta V \> 0\) \\ge 1-\\alpha$$. The loss $$L\_V \= \\mathbb{E}\[\\log(1+e^{-\\Delta V})\]$$. On the event $$\\{\\Delta V \\le 0\\}$$ (prob $$\\le \\alpha$$), the loss is at least $$\\ln 2$$. A simple bound is $$L\_V \\le \\alpha \\cdot \\log(1+e^{2B}) \+ (1-\\alpha) \\cdot \\log(1+e^0) \\approx \\alpha \\ln 2$$ for small $$\\Delta V \> 0$$. As $$\\inf L\_V \\ge 0$$, $$V$$ is $$\\varepsilon\_V$$-optimal for $$\\varepsilon\_V \\approx \\alpha \\ln 2$$.

- Helpful Best Response (H): Let $$\\pi\_Ht$$ be any deviation. The utility gain is bounded by improvements from correctness and verifier score. By (C1), $$\\pi\_H$$ is $$(1-\\alpha)$$-correct, so the max gain on correctness is bounded by $$\\gamma \\alpha$$. By (C2) and (R2), the max gain from verifier score is bounded by $$L \\cdot (2B) \\alpha$$. A combined bound gives the maximum utility gain as $$\\max(\\gamma, 2BL)\\alpha \= \\varepsilon\_H$$.

- Sneaky Best Response (S): The soundness condition (S) ensures that no other sneaky policy $$\\pi\_St$$ can have a probability of success greater than $$\\beta$$. By (R3), the maximum utility gain over $$\\pi\_S$$ is bounded by $$\\delta\\beta$$. If $$\\pi\_S$$ is an additional $$\\varepsilon\_S'$$-best response, the total sub-optimality is $$\\varepsilon\_S \= \\delta\\beta \+ \\varepsilon\_S'$$.

This establishes the equivalence.

### Appendix B \- Sequentiality as Equilibrium Refinement

Under a static analysis with a *fixed* verifier, the distinction between sequential and simultaneous play can appear minimal. However, what we claim is that sequential play can be seen as an equilibrium refinement mechanism, and note how our empirical framework approximates the static, one-shot game, with a dynamic, co-training process for the players of the game over rounds. The sequential structure is crucial precisely because the verifier is *not* fixed; it is actively learning to distinguish the provers.

At a fixed verifier $$V^\*$$, the sneaky prover's optimization problem is indeed:

$$\\max\_{z\_s} V^\*(x, z\_s) \\cdot \\text{success}(x, z\_s)$$

This objective is independent of $$z\_h$$, making the sequential structure seemingly irrelevant at equilibrium. We acknowledge this theoretical observation is valid for fixed verifiers.

### Why Sequentiality Becomes Critical in a Dynamic Game

Once we acknowledge the co-evolution of the verifier, the benefit of sequentiality becomes clear. It changes the nature of the strategic problem for the sneaky prover by altering its information set. Importantly, the sequential structure is not about changing the equilibrium but rather about constraining the path to equilibrium in a way that promotes robust feature learning.

#### Eliminating "Style-Based" Weak Equilibria

- In Simultaneous Play, the honest and sneaky provers develop their strategies in isolation. The honest prover might develop a stylistic signature (e.g., using specific comment formats, variable naming conventions). The sneaky prover can independently discover a *different* stylistic niche. The verifier can then trivially distinguish them by learning these superficial stylistic tells, a phenomenon we call "style leakage." The verifier never has to learn to detect subtle, *semantic* errors. This is a weak, collusive-like equilibrium.

- In Sequential Play, the sneaky prover *observes* the honest solution $$z\_H$$ *before* generating its own $$z\_S$$. For the sneaky prover, adopting a different stylistic signature is now a suboptimal strategy, because the verifier's training objective is to find *any* distinguishing feature. If the sneaky prover produces a $$z\_S$$ that is stylistically different from the observed $$z\_H$$, it provides the verifier with an easy-to-learn signal. A rational sneaky prover, knowing this, is incentivized to mimic the style of $$z\_H$$ perfectly to mask its presence.

#### Distribution Discrimination vs Correctness Classification

Another key point is that this change allows us to shift from correctness classification (a suboptimal requirement, as shown before) to distribution discrimination:

$$L\_V \= \-\\mathbb{E}\[\\log \\sigma(V(x,z\_h) \- V(x,z\_s))\]$$

This trains $$V$$ to distinguish which generative process produced a solution, not whether it's correct. Under simultaneous play, spurious correlations (e.g., “honest provers use more comments”) provide easy discrimination signals, and are weak equilibria the players can converge to[^2]. Sequential play breaks these spurious correlations.

Consider the verifier's hypothesis class at iteration $$t$$. Let $$\\mathcal{F}\_t$$ be the set of features the verifier can use for discrimination.

- Simultaneous Play: $$\\mathcal{F}\_t$$ includes both:
  - Correctness-related features $$\\mathcal{F}\_{\\text{correct}}$$
  - Spurious stylistic features $$\\mathcal{F}\_{\\text{style}}$$
  - The verifier solves
    $$\\min\_V \\mathbb{E}\_{x}\\left\[-\\log \\sigma(V(x, h(x)) \- V(x, s(x)))\\right\]$$
    where $$h, s$$ are fixed best responses to $$V$$.
  - Any function $$V$$ that satisfies $$V(x, h(x)) \> V(x, s(x))$$ for all $$x$$ achieves loss arbitrarily close to 0\. This includes:
    - $$V\_{\\text{robust}}$$: Uses actual correctness/backdoor features
    - $$V\_{\\text{spurious}}$$: Uses stylistic markers (e.g., solution length, comment style)
    - $$V\_{\\text{mixed}}$$: Any convex combination
  - All are valid equilibria\! The simultaneous game provides no pressure to select among them.

- Sequential Play: Since the sneaky prover can copy –without requiring search or heuristics to navigate the strategy space– any observable feature from $$z\_h$$, effectively $$\\mathcal{F}\_{\\text{style}}$$ becomes unusable for discrimination. The verifier is forced to rely on $$\\mathcal{F}\_{\\text{correct}}$$.
  - Now $$s$$ is a function of both $$x$$ AND $$h(x)$$. This breaks the equilibrium multiplicity:
  - \- $$V\_{\\text{spurious}}$$ fails because $$s$$ can copy any stylistic feature from $$h(x)$$
  - \- Only $$V\_{\\text{robust}}$$ remains viable

This is analogous to the revelation principle for auctions in mechanism design (Perry and Reny, 1994). While any equilibrium outcome achievable through indirect mechanisms can be achieved directly, the indirect mechanism (here, sequential play) can have better off-equilibrium properties and be more robust to deviations.

#### Formal Characterization of Equilibria

Let us formally define the equilibrium concepts:

Definition (Feature Space): Let $$\\Phi: \\mathcal{Z} \\to \\mathcal{F}$$ be a feature extraction function where $$\\mathcal{F}$$ is the feature space. We partition $$\\mathcal{F} \= \\mathcal{F}\_{\\text{robust}} \\cup \\mathcal{F}\_{\\text{spurious}}$$ where:

- $$\\mathcal{F}\_{\\text{robust}}$$: Features determined by correctness/backdoor properties
- $$\\mathcal{F}\_{\\text{spurious}}$$: Features independent of correctness (style, length, etc.)

Definition (Copyable Features): A feature $$f \\in \\mathcal{F}\_{\\text{spurious}}$$ is *copyable* if, given any solution $$z\_h$$, there exists a transformation $$T$$ such that $$\\Phi(T(z\_h))$$ preserves $$f$$ while allowing arbitrary modification of correctness.

Lemma 1: For any spurious feature $$f \\in \\mathcal{F}\_{\\text{spurious}}$$ and any $$(x, z\_h)$$ pair, the sneaky prover in sequential play can generate $$z\_s$$ such that:

1. $$\\Phi(z\_s)|\_f \= \\Phi(z\_h)|\_f$$ (matches on feature $$f$$)
2. $$C(x, z\_s) \\neq C(x, z\_h)$$ (differs in correctness)

Proof: By definition of spurious features, $$f$$ is independent of correctness. Therefore, the transformation preserving $$f$$ while flipping correctness exists.

Theorem (Equilibrium Uniqueness): In the sequential game, any equilibrium verifier $$V^\*$$ must satisfy:

$$V^\*(x,z) \= g(x, \\Phi(z)|\_{\\mathcal{F}\_{\\text{robust}}})$$

for some function $$g$$. That is, $$V^\*$$ depends only on robust features.

Proof:

Suppose for contradiction that an equilibrium verifier $$V^\*$$ depends on some spurious feature $$f \\in \\mathcal{F}\_{\\text{spurious}}$$.

Define the honest and sneaky best responses in this purported equilibrium as $$h^\*, s^\*$$.

Case 1: $$h^\*$$ and $$s^\*$$ differ on feature $$f$$ for some $$x$$

For such $$x$$, $$\\Phi(h^\*(x))|\_f \\neq \\Phi(s^\*(x, h^\*(x)))|\_f$$. But by Lemma 1, $$s^\*$$ could deviate to $$\\tilde{s}$$ where:

$$\\Phi(\\tilde{s}(x, h^\*(x)))|\_f \= \\Phi(h^\*(x))|\_f \\qquad \\text{ and }\\qquad C(x, \\tilde{s}(x, h^\*(x))) \= 0$$

Since $$V^\*$$ depends positively on $$f$$ matching $$h\*$$.

Case 2: $$h^\*$$ and $$s^\*$$ agree on feature $$f$$ for all $$x$$. Then $$V^\*$$ cannot use $$f$$ for discrimination. The dependence on $$f$$ is vacuous, contradicting our assumption. Therefore, $$V^\*$$ cannot depend on any spurious feature.

[^1]: In the code domain every spec is satisfiable by construction (tests were written so that at least one reference solution passes).

[^2]:  Theoretically, the sneaky prover has an incentive to deviate and attempt mimicking the honest prover even under simultaneous play. The problem in this case is that i) the space of possible strategies is countably infinite and that ii) empirically –that is, with Neural Networks– convergence to the global optima is not guaranteed, but rather to a local one.