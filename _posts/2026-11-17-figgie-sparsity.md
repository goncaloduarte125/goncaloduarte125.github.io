---
layout: distill
title: "Learning what not to do - Sparsity in Figgie"
description: Sparse policy heads in a hidden-information trading game. We compare softmax, entmax-1.5 and sparsemax against fixed bots, against each other, against trained attackers and inside a shared league.
tags: reinforcement-learning sparse-policies entmax figgie
date: 2026-11-17

authors:
  - name: Gonçalo Duarte
bibliography: 2026-11-17-figgie-sparsity.bib

toc:
  - name: Learning what not to do
  - name: A small market with a hidden payoff
  - name: What is sparse about a sparse policy?
  - name: Two ways to grow the action space
  - name: Agents and training
  - name: Baseline agents
  - name: A first comparison
  - name: Which actions survive?
  - name: How much does a colder softmax buy?
  - name: What happens at the same table?
  - name: Teaching the agent to leave a dollar alone
  - name: What can an opponent learn to take?
  - name: Why did self-play leave these holes?
  - name: Learning against a shared league
  - name: Does the league preserve the trading advantage?
  - name: Conclusion
  - name: Related work
---

> **TL;DR.** Softmax gives every legal action some probability, while sparse probability maps can return exact  *zeros*. We test whether that helps a reinforcement-learning agent trade in Figgie, a four-player card market with hidden information, as the action space grows from 97 to 1,937 actions. Against fixed rule-based opponents, entmax-1.5 earns more than softmax at every market size evaluated, and about \\$4–7 more per game than a cooled softmax against our mixed bot table. Trained attackers still find about \\$19 per game against entmax, and doubling training doesn't fix that. A shared league of policies with all three heads cuts sparsemax's best observed attacker gain to \\$4.1 while entmax keeps its trading lead. Sparse heads win at both ends: entmax is the strongest player trained and sparsemax the hardest to exploit.

## Learning what not to do

Imagine deciding whether to buy a card. You could offer a low price, improve an existing bid, accept someone else's offer, sell something else, or wait. Now give each possible quote dozens of price points. The list of actions grows quickly, even though only a few make sense at any moment.

A policy has to spread its probability over that list. Softmax gives every legal action some chance of being chosen. Sparse policies let the agent rule actions out completely. 

We tested this approach in Figgie, a trading game created by Jane Street that combines incomplete information and negotiation. In this game an action does more than change your holdings. It can also tell your opponents something about what you know. A policy that filters out bad trades might learn faster and perform better while also being easier to read.

## A small market with a hidden payoff

Four players receive ten cards each from a 40-card deck. The suits have **12, 10, 10 and 8** cards, but nobody knows which suit has which count. Players see only their own hands. That leaves twelve possible deck configurations, each equally likely.

Only one suit pays out. The **goal suit** is the other suit of the same colour as the 12-card suit: spades pairs with clubs, hearts with diamonds. If spades is the 12-card suit, clubs is the goal. The goal therefore has either eight or ten cards.

In our version, each player starts with \\$350 and pays a \\$50 ante into a \\$200 pot. At settlement each goal card pays \\$10, and whoever holds the most goal cards takes the rest of the pot: \\$120 for an eight-card goal, \\$100 for a ten-card goal. Tied leaders split it. Other cards pay nothing. Profit is

$$
\text{profit} = \text{final cash} + \text{settlement payouts} - 350,
$$

and without the decoy actions introduced below, the four profits sum to zero.

The bonus makes card values depend on inventory, the same hand can pay differently depending on the playing deck.

### Trading

A game lasts **240 ticks**. On each tick every player picks one action, and actions resolve in a seat order drawn at the start of the game. A player can keep one bid and one ask per suit. A **lift** buys one card at the best ask, and a **hit** sells one card at the best bid. Any completed trade cancels every standing quote at the table, so a quote has a short window to find a counterparty.

A very low bid is great if someone accepts it and useless if nobody does. A generous bid trades immediately but gives away value. Learning to trade means balancing price, chance of acceptance, inventory, time and information.

Each agent observes its own hand, public cash and quotes, and public transfers. It also receives a Bayesian posterior over the twelve deck configurations, computed from the card counts it knows about. This gives every agent the same starting point for card inference. It does not treat opponents' strategic choices as evidence, and it does not reveal their hands.

## What is sparse about a sparse policy?

A policy maps the current observation to a distribution over actions. Given scores $\mathbf z$ for the legal actions, softmax can be written as an entropy-regularised argmax over the probability simplex $\Delta$:

$$
\operatorname{softmax}(\mathbf z) = \arg\max_{\mathbf p \in \Delta} \; \mathbf p^\top \mathbf z + H^{S}(\mathbf p),
\qquad H^S(\mathbf p) = -\textstyle\sum_a p_a \log p_a .
$$

Replacing Shannon entropy with Tsallis entropy gives $\alpha$-entmax<d-cite key="peters2019sparse"></d-cite>:

$$
\alpha\text{-entmax}(\mathbf z)_a = \big[(\alpha-1) z_a - \tau\big]_+^{1/(\alpha-1)},
\qquad \tau \text{ s.t } \textstyle\sum_a p_a = 1 .
$$

As $\alpha \to 1$ this recovers softmax, and $\alpha = 2$ is sparsemax<d-cite key="martins2016softmax"></d-cite>. For any $\alpha > 1$, every action whose score falls below the threshold gets **exactly zero** probability. The actions left with positive probability form the policy's **support**, and the support can change from one state to the next.

| Possible action in one situation | Illustrative dense policy | Illustrative sparse policy |
| --- | ---: | ---: |
| Bid near the estimated value | 60% | 65% |
| Accept an attractive ask | 25% | 25% |
| Wait | 10% | 10% |
| Offer an implausibly low bid | 4% | 0% |
| Buy at an unattractive price | 1% | 0% |

*Invented distributions, shown to illustrate the distinction. They are not measured policies.*

Both policies randomise. The difference is whether every legal action stays possible. This is not greedy action selection, and it is not rule enforcement: every agent is already masked from illegal actions, such as selling a card it does not hold. The question is whether to exclude actions that are legal but currently unattractive.


### How far can regularisation pull a policy from optimal?

The toy example has a theoretical counterpart<d-cite key="gaehtgens2024sparse"></d-cite>. An entropy-regularised agent maximises reward plus an entropy bonus, so its optimal policy is generally not the reward-optimal one: it gives up some return in exchange for spread. How much it can give up is bounded by how much entropy is available to buy.

Write the Tsallis negentropy as $\Omega_\alpha(\boldsymbol\pi) = \tfrac{1}{\alpha}\,\mathbb E_{a\sim\boldsymbol\pi}\big[\log_{2-\alpha} \pi_a\big]$, with $\log_\beta u = (u^{1-\beta}-1)/(1-\beta)$ and $\log_1 = \log$, so $\alpha = 1$ is Shannon. It is zero for a deterministic policy and most negative for the uniform one. With regularisation weight 1 and discount $\gamma$, the return $J$ of the optimal regularised policy $\pi^\star_\alpha$ satisfies

$$
J(\pi^\star) + \frac{1}{\alpha(1-\gamma)} \log_{2-\alpha}\!\Big(\frac{1}{|A|}\Big) \;\le\; J(\pi^\star_\alpha) \;\le\; J(\pi^\star),
$$

where $\pi^\star$ is the unregularised optimum. The width of this window is the worst-case **performance gap**:

$$
U - L = \frac{1}{1-\gamma} \cdot
\begin{cases}
\log |A|, & \alpha = 1 \text{ (softmax)},\\[4pt]
\dfrac{1 - |A|^{1-\alpha}}{\alpha(\alpha-1)}, & \alpha > 1 .
\end{cases}
$$

The two cases behave very differently as $\lvert A \rvert$ grows. Shannon entropy of the uniform distribution is $\log \lvert A \rvert$, which has no ceiling, so each extra action gives a softmax agent a little more entropy to collect by spreading mass, and the bound on what that can cost grows without limit. For $\alpha > 1$ the term $\lvert A \rvert^{1-\alpha}$ vanishes and the gap levels off at

$$
\lim_{|A|\to\infty} (U - L) = \frac{1}{\alpha(\alpha-1)(1-\gamma)} .
$$

Tsallis entropy with $\alpha > 1$ is bounded no matter how many actions there are, so past some point new actions buy almost no extra entropy and there is little reason to put mass on them. As $\alpha \to 1$ this limit itself diverges, which is the softmax case again.

Per decision (dropping the $1/(1-\gamma)$ factor), at the sizes in our sweeps:

| Actions $\lvert A \rvert$ | Softmax ($\alpha=1$) | Entmax-1.5 | Sparsemax ($\alpha=2$) |
| ---: | ---: | ---: | ---: |
| 97 | 4.57 | 1.20 | 0.49 |
| 257 | 5.55 | 1.25 | 0.50 |
| 497 | 6.21 | 1.27 | 0.50 |
| 977 | 6.88 | 1.29 | 0.50 |
| 1,937 | 7.57 | 1.30 | 0.50 |
| $\to\infty$ | $\to\infty$ | 1.33 | 0.50 |

Going from 97 to 1,937 actions, the softmax gap grows by 3.0 while entmax-1.5's grows by 0.1.

This should be read with care. It is a worst-case bound, not a prediction of how much a given agent loses. It describes the exact optimum of a regularised, fully observed MDP, not a PPO agent in a partially observed game against adapting opponents. Our games are episodic with $\gamma = 1$, where the $1/(1-\gamma)$ factor becomes the number of decisions. And the scale is set by the regularisation weight, here fixed at 1: shrinking a softmax agent's entropy weight like $1/\log \lvert A \rvert$ would keep its gap constant too. That is one reason the temperature-matched softmax controls below matter.

### Why trading makes it harder

In a single-agent task the environment does not interpret your choices. In Figgie the other players can. If your quotes concentrate on buying clubs after your hand makes clubs look valuable, an observant opponent can infer that and raise their price. Sparse policies might help focus on profitable actions but they might also reveal information and make agents more exploitable. 

## Two ways to grow the action space

All actions share one categorical distribution. With $P$ price levels and $K$ decoys,

$$
|A| = 1 + 8P + 16 + K .
$$

That is one no-op, a bid and an ask for each of four suits at each price, four actions each for cancelling bids, cancelling asks, lifting and hitting, and the decoys. Prices run from $30/P$ to \\$30 in steps of $30/P$.

A larger action count can mean more useful choices or just more clutter, so we grow it in two different ways.

**Finer prices (resolution sweep).** Increase $P$ over the same price range. This adds real choices and changes the market's granularity.

**Useless actions (padding sweep).** Hold $P = 30$ and add $K$ decoys. A decoy does nothing except burn \\$1 of the acting player's cash, so waiting always dominates it. The useful market stays the same, and mistakes show up directly as money burned.

Decoys helps us understand how good the agents are at avoiding strictly dominated actions and discerning between cases where an agent is better at dealing finer price granularity vs a strictly larger action space.

| Sweep | Price levels $P$ | Price step | Decoys $K$ | Actions $\lvert A \rvert$ |
| --- | ---: | ---: | ---: | ---: |
| Resolution | 10 | \\$3.00 | 0 | 97 |
| Resolution | 30 | \\$1.00 | 0 | 257 |
| Resolution | 60 | \\$0.50 | 0 | 497 |
| Resolution | 120 | \\$0.25 | 0 | 977 |
| Padding | 30 | \\$1.00 | 240 | 497 |
| Padding | 30 | \\$1.00 | 720 | 977 |
| Padding | 30 | \\$1.00 | 1,680 | 1,937 |

The two sweeps meet at the same sizes but ask different questions. At 497 actions, one market offers half-dollar prices and the other keeps dollar prices plus 240 costly distractions. Calling both "a larger action space" would hide the difference.

## Agents and training

Every agent uses the same actor, critic, demonstrations, PPO procedure and opponent pool. Only the policy head changes: the map from logits to probabilities and its matching entropy bonus.

| Arm | Policy head | Purpose |
| --- | --- | --- |
| Softmax | Dense over legal actions | Baseline |
| Entmax-1.5 | Sparse, $\alpha = 1.5$ | Main sparsity comparison |
| Sparsemax | Sparse, $\alpha = 2$ | Stronger sparsity |
| Anneal | $\alpha$ rises from 1 to 2 during training | Start dense, finish sparse |
| Entmax-1.5 + $\epsilon$ | 98% entmax + 2% uniform over legal actions | Keep every legal action reachable |

The entropy bonus is Shannon entropy for softmax and the matching Tsallis entropy for the sparse heads,

$$
S_\alpha(\mathbf p) = \frac{1 - \sum_a p_a^\alpha}{\alpha(\alpha - 1)},
$$

with Shannon entropy as the $\alpha \to 1$ limit. The two bonuses have different scales, so one coefficient would not give equal regularisation. We tune $\lambda \in \{0.003, 0.01, 0.03\}$ separately for each arm on two tuning seeds, kept apart from the five reporting seeds.

**Network.** The actor takes a 69-dimensional observation through two 256-unit ReLU layers and a 128-unit GRU, and outputs one logit per action. A separate critic sees privileged information, including all hands and the true deck. The actor never does.

**Warm start.** Each agent first imitates a Bayesian bot on 500 demonstration games for ten epochs, with one game per gradient step. We use a Fenchel–Young loss<d-cite key="blondel2020learning"></d-cite>, which stays well defined when a sparse head gives the expert's action zero probability. For softmax it reduces to cross-entropy. 

**PPO.** About 50M environment steps (1,024 parallel games, 128-tick rollouts, 381 updates), learning rate $2.5 \times 10^{-4}$ with linear decay, clip 0.2, $\gamma = 1$, GAE $\lambda = 0.95$. The probability ratio is computed directly, so a new probability of zero is allowed. Opponents come from a pool of 16 frozen snapshots chosen by prioritised fictitious self-play, with weights proportional to $(1 - w)^2$ for estimated win rate $w$. Rule-based bots fill 30% of opponent seats.

**Reward.** The base reward is terminal profit in dollars. Training adds potential-based shaping with $\Phi(s)$ equal to cash plus the posterior value of held cards:

$$
r_t = \mathbf 1_{\text{terminal}}\,(W_T - 350) + 0.5\,\big(\Phi(s_{t+1}) - \Phi(s_t)\big).
$$

The terminal potential is zero, so at $\gamma = 1$ the shaping telescopes to a constant. All evaluation reports **unshaped dollar profit**.

## Baseline agents

### Rule-based bots

Before training anything, we validate the baseline using 4 heuristic bots, playing at $P = 30$, $K = 0$. The Bayesian bot follows the Bayesian fundamentalist of DiSilvio et al.<d-cite key="disilvio2021traders"></d-cite>

| Bot | Strategy |
| --- | --- |
| Bayesian | Updates its deck posterior as cards become known and prices in the pot bonus |
| FairValue | Trades around the expected \\$10 goal payout from its initial hand, with a \\$2 edge |
| MarketMaker | Quotes around initial-hand expected values with a \\$3 half-spread |
| Conservative | Uses a \\$4 edge and waits until its top goal-suit probability reaches 0.4 |

Over 20,000 games, with one bot of each type and resolution order randomised per game:

| Bot | Mean profit (\\$/game) | Win rate |
| --- | ---: | ---: |
| Bayesian | +19.11 | 59.8% |
| FairValue | +0.45 | 17.1% |
| MarketMaker | −8.02 | 12.8% |
| Conservative | −11.54 | 10.3% |

The ranking matches the one in Yang's CS224R project<d-cite key="yang2026figgie"></d-cite>. The numbers are not directly comparable, because that tournament also seated a Random agent whose losses become other bots' gains. Wealth is conserved at \\$1,400 per table, as it should be without decoys.

### A learned softmax baseline

Next we check that the shared recipe learns a profitable policy before we change the head. Three softmax runs at $P = 30$, $K = 0$ and $\lambda = 0.01$, evaluated over 10,000 games per matchup with sampled actions, earn **\\$8.7–10.2 per game** against FairValue + Bayesian + MarketMaker and **\\$13.1–14.1** against three Bayesian bots. These numbers can't be ranked against the tournament table: there the four seats hold four different bots, while here the learned agent takes one seat against fixed opponents.

Opponent mix matters even for this baseline. A seed-0 run trained with a mid-training opponent swap reaches \\$16.11 against the mixed pool but only \\$3.35 against three Bayesian bots, compared with \\$10.17 and \\$13.06 without the swap. Doing better against one table can mean doing worse against another, so swap experiments stay out of the head comparison.

## A first comparison

The main grid trains each head at all seven sizes with five reporting seeds. Each final checkpoint takes one seat in two fixed matchups: against FairValue, Bayesian and MarketMaker, and against three Bayesian bots. Both heads sample actions at temperature 1.

{% include figure.liquid loading="eager" path="assets/img/figgie_rl/baselines_softmax_vs_entmax.png" title="Softmax vs entmax-1.5 against fixed baseline opponents. Bars are medians over five seeds; dots are individual seeds. Left of each divider: finer prices, no decoys. Right: 30 prices plus $1 decoys. Win rate counts games where the agent finishes with the most wealth, ties split, so 25% is par." class="img-fluid rounded z-depth-1" %}

Entmax-1.5 earns more than softmax at every size, in both matchups. Some medians:

| Size | Actions | vs mixed pool: softmax | vs mixed pool: entmax-1.5 | vs 3 × Bayesian: softmax | vs 3 × Bayesian: entmax-1.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 prices | 97 | \\$7.3 | \\$12.8 | \\$8.3 | \\$12.9 |
| 30 prices | 257 | \\$10.2 | \\$17.3 | \\$14.9 | \\$19.9 |
| 120 prices | 977 | \\$13.4 | \\$20.2 | \\$17.6 | \\$24.2 |
| +1,680 decoys | 1,937 | \\$10.0 | \\$16.7 | \\$13.0 | \\$15.7 |

Two patterns are worth noting.

- **Finer prices help both heads.** Profit rises from 10 to 120 price levels for softmax and entmax alike, so the extra resolution is useful, not just clutter.
- **Win rate tells a different story from profit.** Against the mixed pool, softmax wins about 38–40% of games at every size. Entmax goes from 41% at 10 prices to 48% at 120 and about 53–54% with decoys. Against three Bayesian bots, softmax stays below par (about 18–20%) while entmax sits near or above it (23–32%).

This is a comparison against fixed, rule-based opponents. It does not yet tell us *why* entmax does better. Softmax at temperature 1 may simply be too diffuse, and a cooler softmax could close much of the gap. Nor does it tell us how either policy holds up against opponents that adapt to it.

The rest of this post takes those questions in turn: whether the agent is filtering or just more decisive, whether a simpler concentration control explains the gap, what happens when the heads share a table, and how exploitable sparse policies are.

## Which actions survive?


The first comparison found that entmax-1.5 earned more than softmax against fixed bots at every market size. To understand that result, start with the ordinary market of 30 prices and no decoys. Of its 257 actions, only some are legal in a given state. We can count how many receive positive probability (**support**) and measure the probability of the single most likely action (**concentration**).

| Policy | Mean legal choices | Choices with positive probability | Probability of favourite action |
| --- | ---: | ---: | ---: |
| Softmax | 157 | All legal choices | 7.3% |
| Entmax-1.5 | 167 | 33 | 25.4% |
| Sparsemax | 166 | 31 | 27.3% |

*Each entry is a median across five training seeds after averaging the final 20 logged updates within each seed. Each policy visits its own states, so even the number of legal choices differs. The softmax logger counts about 95 actions above probability 0.001. Its mathematical support includes the rest.*

The sparse heads both filter and concentrate. Entmax assigns zero probability to about four in five legal actions and still samples something other than its favourite roughly three quarters of the time. The rule that produces those zeros depends on the actions' learned scores<d-cite key="peters2019sparse"></d-cite>. It offers no guarantee about *which* actions survive. We can test how much the learned ranking matters by keeping softmax's ranking intact and changing how often it samples from the tail.

## How much does a colder softmax buy?

A temperature of $T=0.35$ divides every logit by 0.35 before sampling. A bid that already outranked an ask keeps its lead, and the odds tilt further towards the bid. We chose the temperature on two held-out training seeds at 30 prices, using profit against one FairValue, one Bayesian and one MarketMaker bot. We then froze that choice and evaluated seeds 0–4, with 10,000 games per checkpoint and matchup. Entmax stayed at its training temperature of 1.

At 30 prices, cooling takes softmax from \\$10.2 to \\$12.5 per game against the mixed table. Entmax earns \\$16.7. To see how often the difference survives a change of training seed, subtract cooled softmax's return from entmax's return *within each seed*. The median of those five differences is **+\\$4.4**. Now keep the checkpoints and deal them in against three Bayesian bots. The same comparison gives **+\\$0.6**.

{% include figure.liquid loading="eager" path="assets/img/figgie_rl/temperature_gap_by_opponent.png" title="Entmax-1.5 at temperature 1 versus softmax at 0.35. Each dot is a paired training seed and each diamond is the median. Positive differences favour entmax. Rows below the divider keep 30 prices and add decoys." class="img-fluid rounded z-depth-1" %}

Against the mixed table, all 35 paired differences are positive. Against three Bayesian bots, softmax pulls ahead at all three decoy sizes. The clearest remaining entmax advantages there come from finer prices, about \\$2.8 per game at 60 price levels and \\$4.5 at 120. The temperature was selected for the mixed table, so the Bayesian result required no extra tuning for that opponent.

We also tried cutting off softmax's tail outright. Keeping the smallest set of actions containing 95% of its probability mass, then sampling from that set, closes roughly 10–30% of the original mixed-table gap. Training a softmax with truncation also leaves it behind plain entmax at all seven sizes. These controls show how much we can recover by changing action selection. They leave open how much of the remaining gap comes from the zeros themselves, the training gradients, or the different market situations the agents encountered while learning.

## What happens at the same table?

So far the learned traders have occupied separate games. Seat one entmax trader with three softmax traders and both sides now get to accept the other's prices. Then change the mix until the *softmax* trader is the one surrounded by rivals. We score each table by mean profit **per entmax seat minus mean profit per softmax seat**. This keeps a three-seat side from getting triple credit for its trades.

| Market | 1 entmax seat, 3 softmax | 2 of each | 3 entmax seats, 1 softmax |
| --- | ---: | ---: | ---: |
| 30 prices | +\\$22.9 | +\\$15.8 | +\\$10.4 |
| 120 prices | +\\$22.6 | +\\$12.2 | +\\$6.4 |
| 30 prices + 240 decoys | +\\$14.8 | +\\$1.9 | −\\$5.7 |
| 30 prices + 720 decoys | +\\$0.9 | −\\$12.0 | −\\$12.4 |
| 30 prices + 1,680 decoys | +\\$8.7 | −\\$13.8 | +\\$0.4 |

*Median per-seat gaps in dollars per game over five seeds, plain entmax-1.5 at $T=1$ against softmax at $T=0.35$. A positive gap counts as one positive seed × table result, regardless of how many individual games either policy won.*

Across the four price grids without decoys, entmax leads in **60 of 60** comparisons (four grids × three seat mixes × five seeds). Adding 720 decoys produces a striking reversal. A lone entmax seat is close to level at the median. Put two copies at the table and each trails a softmax seat by \\$12. The training seed matters here too. At that two-against-two table the 95% seed-bootstrap interval runs from −\\$17.3 to +\\$13.1.

Where does the money go? Separate replays with both policies at $T=1$ let us account for the initial deal, gains on trades at realised card payouts, the change in majority bonus, and dollars burned. In the ordinary 30-price market, a lone entmax seat gains a median **\\$16.8 in majority bonus** while giving up **\\$5.9 on trades**. Paying more than a goal card's eventual \\$10 payout can make sense if that card also tips the majority.

With three entmax seats and decoys, the lone softmax trader finds quotes worth accepting. In the six seed-and-size cases where the entmax seats lose, softmax gains \\$17–26 per game from taking quotes, valued at the cards' realised payouts. Entmax's own decoy burn accounts for 34–75% of its per-seat loss in those cases. Those dollars leave the table entirely. Softmax collects its trading gains when it accepts a price posted by another player.

## Teaching the agent to leave a dollar alone

A decoy is always worse than waiting. Why would a policy that can discard most actions spend money on one? The Bayesian demonstrator never picks a decoy. During behaviour cloning, once a sparse head assigns an undemonstrated action zero probability, the imitation loss supplies no further direct gradient on that score<d-cite key="blondel2020learning"></d-cite>. In one reduced-scale replay, excluded decoys finished imitation only 0.17 logits below the threshold for entering support. Subsequent updates to the shared network could bring them back. That distance comes from a single pilot. Swapping entropy coefficients in the full runs did little to change the burn.

Our first fix pushed every currently excluded action farther below the threshold during imitation. It eliminated entmax's decoy burn and also taught the trader to stop accepting offers. Even with *zero* decoys, its mixed-table profit went from +\\$17.3 to **−\\$13.1**. An action the policy excludes today can be the expert's best move in another hand.

The revised loss applies that extra margin only to actions the expert **never took anywhere in the demonstrations**. It needs no decoy labels. The contrast is easiest to see by seating four copies of one frozen policy together. With no outside trader to win money from, the four players' combined loss is exactly the amount they burn.

| Dollars destroyed per four-copy table per game | 240 decoys | 720 decoys | 1,680 decoys |
| --- | ---: | ---: | ---: |
| Softmax | \\$0.9 | \\$2.1 | \\$3.7 |
| Entmax-1.5 | \\$9.5 | \\$12.4 | \\$13.6 |
| Entmax-1.5 with revised margin | \\$0.0 | \\$0.0 | \\$0.3 |

*Five-seed medians rounded to the nearest tenth. These are table totals in frozen-policy evaluation. Training logs include changing opponents and give different burn figures.*

With the revised warm start, entmax leads cooled softmax in **45 of 45** padded seed × table comparisons, up from **23 of 45**. There is a trade-off elsewhere. Against the mixed bot table the revised policy earns about \\$2–3 less per game at 240 and 720 decoys, where plain entmax had earned more than it did in the unpadded market. Why adding useless buttons raised that particular score remains open.

## What can an opponent learn to take?

Apart from evaluating their performance, we want to understand what happens when we train an attacker against a specific agent, i.e how exploitable is the learned strategy. For each plain target seed we try four attackers, including one initialised from the target, and keep the highest measured profit. The table reports the median of those five seed-level best scores.

| Frozen target market | Attacker vs softmax | Attacker vs entmax-1.5 | Attacker vs sparsemax |
| --- | ---: | ---: | ---: |
| 30 prices | +\\$13.7 | +\\$23.2 | +\\$13.1 |
| 120 prices | +\\$6.7 | +\\$20.0 | +\\$4.3 |
| 30 prices + 1,680 decoys | +\\$8.6 | +\\$21.4 | +\\$5.0 |

*Dollars earned by the single attacker per game. A higher score means a more profitable response was found. Each measured return has sampling error, and selecting the highest of several estimates can push it upwards.*

The earlier softmax attacker earned about zero against softmax at 30 prices. A stronger response earned \\$13.7 against the same target family. An attacker's achievable expected profit gives a lower bound on what an optimal response could earn, so a weak training run can make a target look unusually safe. The \\$5.0 figure for padded sparsemax deserves particular care. Its warm-started attacker finished below zero against three of five target seeds.

The sharpest test of the decoy explanation is the repaired entmax policy. At 1,680 decoys its four-copy table burns only \\$0.3, yet the best measured response among three attacker variants still earns **\\$19.4 per game** (95% seed-bootstrap interval \\$15.9–25.0). An opponent can make good money at this table even after the conspicuous dollar-burning mistake has largely disappeared.

## Why did self-play leave these holes?

Every policy above already trained against earlier versions of itself. In each game, 30% of opponent seats came from a five-bot zoo and 70% from a ring of sixteen frozen snapshots. The snapshots all belonged to one training lineage and used that learner's head. An entmax learner could improve against its own past without ever meeting a learned softmax trader.

The next experiments ask whether the remaining weakness comes from cycling, too little training, or the opponents in that ring. All use **30 prices and zero decoys**. They therefore test the vulnerability that survives even when burning money is impossible. The predictions were recorded before the corresponding phases ran.

First, take ten checkpoints along each lineage and play every pair at a two-against-two table. Among pairs whose payoff gap exceeds two standard errors, the later checkpoint wins 650 of 675 comparisons at 50M steps and 667 of 675 at 100M. None of roughly 3,300 fully decided triples forms a cycle. The policies keep improving against their predecessors. The predicted rock-paper-scissors pattern fails to appear in these sampled checkpoints.

Attacker profit also falls sharply early in training. For entmax it drops from about \\$107 at update 50 to \\$18 at the 50M endpoint. That suggests a simple possibility. Perhaps the original budget stopped entmax just before it became difficult to exploit.

| Target | Best attacker at 50M steps | Best attacker at 100M steps |
| --- | ---: | ---: |
| Softmax | \\$8.4 | \\$8.5 |
| Entmax-1.5 | \\$17.8 | \\$19.5 |
| Sparsemax | \\$14.6 | \\$12.0 |

*Median attacker profit per game over three target seeds, using the best of two attackers in both columns. These runs have narrower attacker coverage than the earlier four-attacker comparison. The 100M runs also stretch the learning-rate schedule and change the fraction of training history retained by the snapshot ring.*

Twice the training leaves entmax's final attacker profit close to its earlier level. The difference between entmax and softmax remains \\$11.0 per game, with a 95% seed-bootstrap interval of \\$3.7–13.5. More training does help all three heads against the all-Bayesian table, by about \\$3–5 per game, but it does little to close this measured vulnerability.

One long entmax run illustrates why those outcomes can diverge. Its mixed-bot return jumps from roughly \\$19 to \\$40 as its offers start being accepted much more often. At a checkpoint near the emergence of that behaviour, a warm-started attacker earns about **\\$200 per game**. By the final checkpoint that attacker gain is back near \\$20. A policy can discover a lucrative way to trade with the bots while briefly becoming much easier for a trained rival to exploit.

Both bot evaluations use bot types present during training. Their tables provide controlled comparisons, but generalisation to unseen learned opponents needs its own measurement.

## Learning against a shared league

We tried two ways to broaden the opposition. The first stores each snapshot with a randomly selected softmax, entmax or sparsemax head. It changes how the snapshot's learned scores become probabilities. The second introduces eighteen finished policies trained with all three heads on separate tuning seeds. Every learner gets the same league members, each using its original head. Training seats become 30% bots, 35% league members and 35% the learner's own snapshots.

The three conditions received the same four attacker variants. This comparison uses five target seeds and a new plain-PFSP control set, so its baseline values differ from the earlier table.

| Target head | Plain PFSP | Random snapshot head | Shared league |
| --- | ---: | ---: | ---: |
| Softmax | \\$11.9 | \\$14.1 | \\$16.3 |
| Entmax-1.5 | \\$20.2 | \\$29.7 | \\$17.5 |
| Sparsemax | \\$13.3 | \\$19.7 | **\\$4.1** |

*Median best observed attacker profit per game. For each target seed, take the highest profit among four attackers, then take the median across targets.*

Randomising the snapshot head raises the median attacker profit for every arm. Sparsemax's increase of \\$6.3 has a 95% seed-bootstrap interval of \\$0.3–12.3. The corresponding changes for softmax and entmax have intervals spanning zero. Altering how existing weights are read provides little evidence of the strategic diversity we wanted.

The shared league produces a different result. Sparsemax's best observed attacker gain falls by **\\$9.2**, with an interval of \\$5.1–12.6 for the reduction. Entmax improves by only \\$2.7, with an interval spanning zero. Its gap with softmax narrows largely because the softmax median rises. The intervention designed to explain entmax's weakness has its clearest effect on sparsemax.

Attacker choice remains consequential. Across these conditions, the entmax attacker is best against softmax in 14 of 15 target cases. The two-attacker experiments above omit that attacker and understate softmax's measured vulnerability. The 100M results establish what those two attackers found. Matching the full four-attacker coverage would make the training-budget comparison more informative.

## Does the league preserve the trading advantage?

The league costs entmax about \\$3.9 per game against the mixed bots and sparsemax about \\$3.1. We can now ask whether that cost also appears against learned traders that were absent from the league.

Each learner takes one seat against three copies of a held-out main-grid policy. Every learner seed plays every opponent seed, giving 25 seed pairs per cell and 10,000 games per pair.

| League learner | Against held-out softmax | Against held-out entmax | Against held-out sparsemax |
| --- | ---: | ---: | ---: |
| Softmax | +\\$1.4 | −\\$7.9 | −\\$6.5 |
| Entmax-1.5 | +\\$13.9 | −\\$0.7 | +\\$4.9 |
| Sparsemax | +\\$8.9 | −\\$1.5 | +\\$1.5 |

*Mean profit per learner seat, averaged across seed pairs. These figures measure the learner's own profit. The direct contests below use the difference between the two sides' per-seat profits.*

Entmax earns the most against every held-out head. Against plain policies of their own head, league learners stay close to level. A separate direct comparison with the plain PFSP controls finds that league softmax and league sparsemax improve by about \\$2 per seat, while entmax's change is unresolved. The lower bot score therefore accompanies preserved or improved performance in these learned-policy evaluations.

Finally, put the league-trained agents at the same table. Average the per-seat gap across one-against-three, two-against-two and three-against-one compositions, with all 25 seed pairs.

| Matchup | Gap at temperature 1 | Gap with softmax cooled to 0.35 |
| --- | ---: | ---: |
| Entmax-1.5 minus softmax | +\\$15.6 | +\\$12.3 |
| Sparsemax minus softmax | +\\$10.9 | +\\$6.2 |
| Entmax-1.5 minus sparsemax | +\\$2.4 | — |

Cooling narrows the gaps, but entmax still leads in all 25 seed-pair averages and sparsemax in 24. Entmax's cooled gap has a 95% interval of \\$8.7–16.1. The temperature comes from the earlier plain-softmax tuning experiment. It could be improved by tuning specifically for league softmax against learned opponents.

Between the sparse heads, entmax wins 21 of 25 seed pairs and leads by \\$2.4, with an interval of \\$0.6–4.4. Yet its best observed attacker earns \\$17.5, compared with \\$4.1 against league sparsemax. A policy's success against these learned traders and the opportunity available to a specialised attacker give different rankings.

The \\$4.1 result is promising, and also a reason to strengthen the attacks. Both softmax-initialised attackers lose money against league sparsemax at the median. Larger attacker budgets could reveal opportunities they missed. These experiments establish a reduction in the best profit found under the tested attack procedure. Exact exploitability remains unknown.

## Conclusion

We asked whether sparse policies can help RL agents perform better in incomplete information games. Across the experiments, the answer is that sparsity pays on both fronts we measured. The strongest trader we trained has a sparse head, and so does the hardest one to exploit. Softmax is neither.

**Entmax-1.5 is the strongest trader.** It earns more than softmax against fixed bots at every market size, in both matchups. Cooling softmax to its best tuned temperature narrows that gap without closing it: all 35 paired seed differences against the mixed table favour entmax. When the two heads share a table, entmax leads in 60 of 60 comparisons without decoys, and once the decoy failure is repaired it also leads in 45 of 45 padded comparisons. After league training, entmax earns the most against held-out policies of every head, beats league softmax by \\$15.6 per seat (\\$12.3 with softmax cooled) in all 25 seed pairs, and edges out league sparsemax by \\$2.4.

**Sparsemax is the least exploitable.** In the first attack comparison, the best attacker against sparsemax earned less than the best attacker against softmax in every market we tried: \\$13.1 against \\$13.7 at 30 prices, \\$4.3 against \\$6.7 at 120 prices and \\$5.0 against \\$8.6 with 1,680 decoys. The shared league widens that margin. League sparsemax gives its best observed attacker \\$4.1 per game, against \\$16.3 for league softmax and \\$17.5 for league entmax, and it is the only head where the league produced a clearly resolved reduction. That robustness does not come at the cost of trading: league sparsemax still beats league softmax by \\$10.9 per seat, or \\$6.2 with softmax cooled, leading in 24 of 25 seed pairs.

Together these results point to the same mechanism from two sides. Ruling actions out lets a policy concentrate on the trades that matter, which is where entmax's edge comes from. Pushing sparsity further, and training against a diverse league, leaves an opponent fewer stray quotes to pick off, which is where sparsemax's robustness comes from. The exponent $\alpha$ moves a policy between those two ends, and both ends beat a dense policy.

The benefits come with conditions. Sparse heads need a warm start that does not leave undemonstrated actions just below the support threshold, or they burn money on actions that are never worth taking. Entmax's best attackers still find about \\$17–20 per game, and more self-play alone does not remove that. Attacker profit is a lower bound, so the \\$4.1 figure for sparsemax shows that our attacks found little, not that little is there. Within those limits, sparse policies gave us both the best trader and the most robust one in this market.

## Related work

**Figgie.** DiSilvio, Luo and Ozerov<d-cite key="disilvio2021traders"></d-cite> study Figgie with agent-based market simulation. Their Bayesian fundamentalist is the basis for our Bayesian bot and for the demonstrations used in the warm start. Yang<d-cite key="yang2026figgie"></d-cite> brings recurrent reinforcement learning to the game and treats negotiation as a core skill: a good offer balances value against its chance of being accepted. That work is the starting point for our simulator and training setup. We adapt it to compare policy heads, and this is not an exact reproduction.

**Sparse probability maps.** Sparsemax<d-cite key="martins2016softmax"></d-cite> replaced softmax with a Euclidean projection onto the simplex, producing exact zeros while remaining trainable end to end. $\alpha$-entmax<d-cite key="peters2019sparse"></d-cite> interpolates between softmax and sparsemax through Tsallis entropy. Fenchel–Young losses<d-cite key="blondel2020learning"></d-cite> pair each such map with a matching convex loss, which we use for behaviour cloning.

**Sparse policies in RL.** Gaehtgens<d-cite key="gaehtgens2024sparse"></d-cite> learns sparse policies with Tsallis entropy regularisation and studies them in large discrete action spaces, which motivates our sweeps. That analysis concerns regularised MDPs. It does not establish that sparsity helps in a partially observed market with strategic opponents, and Figgie is a good place to try it.
