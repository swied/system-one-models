# Model bakeoff: what the results mean

D1 gave the strongest answers to choice and yes/no questions in this test. Jev
was faster than D1 and slightly better at rating things such as urgency. Laya
performed worst on answer quality, but it was the fastest and the only model
that ran locally. Those differences give us three useful options to investigate,
depending on whether a product needs better answers, shorter waits, or local
processing.

This report describes the real-model run saved in
[model-bakeoff.ipynb](../notebooks/model-bakeoff.ipynb).
Our measurements come from that notebook, not from vendor benchmarks. Internet
references below support the explanation of how these models work and what we
could try next; they do not independently verify our results.

## What we tested

Each model received the same 18 example situations and answered 33 questions.
There were 13 questions asking it to choose an option, 14 asking for a yes/no
judgment, and six asking for a rating. Examples included routing a support
request, recognizing a suspicious email, assessing urgency, and deciding whether
a proposed action needed review. The expected answers are recorded in
[prompts.json](../src/system_one_models/prompts.json).

This is useful because many product features need a small decision rather than
a paragraph of generated text. A support tool might need a team name and an
urgency rating before anyone writes a reply. Liquid's documentation describes
these models as tools for classification, routing, and scoring, and explains
how several questions can be answered in one request.
[Source: Liquid's decision-model documentation](https://docs.liquid.ai/lfm/models/decision-models).

## The results in everyday terms

| What we measured | Jev | D1 | Laya |
| --- | --- | --- | --- |
| Correct choice answers | 12 of 13 (92.3%) | 13 of 13 (100%) | 9 of 13 (69.2%) |
| Correct yes/no answers | 12 of 14 (85.7%) | 14 of 14 (100%) | 10 of 14 (71.4%) |
| Average distance from the expected rating, in rating steps | 0.370 | 0.396 | 0.896 |
| Typical response time | 119.9 ms | 308.6 ms | 24.1 ms |
| Where inference ran | Hosted service | Hosted service | Local machine |
| Failed or unusable answers | 0 | 0 | 0 |

Source: the saved notebook output. Answer counts are calculated from the displayed
percentages and the question counts in the scenario file. A lower rating error
means the model stayed closer to the expected answer. “Typical response time”
means the middle recorded time, also called the median.

D1 matched all 27 expected choice and yes/no answers. Jev matched 24, and Laya
matched 19. These totals leave out the rating questions, which are measured by
how far the returned number is from the expected rating. D1's perfect result on
those 27 questions does not mean it answered every kind of question perfectly.

Jev's average rating error was slightly lower than D1's. Both were below half a
rating step; Laya was close to a full step away on average. For a product that
prioritizes work by urgency, that distance matters: moving something between
“normal” and “urgent” can change which customer gets attention first. This is a
product interpretation of the measured rating errors, not evidence that a
particular customer was affected.

## Why you should care

### Better answers can mean less rework

Choosing the wrong support team can create another handoff. Missing a suspicious
message can expose a user to a bad experience; flagging a legitimate message can
interrupt useful work. These are examples of why the cost of a mistake depends
on the feature, even when the model produces an answer successfully.

My recommendation from this run is to make D1 the first candidate for a broader
trial of choice and yes/no tasks, with Jev as a strong alternative. Jev deserves
particular attention when the feature needs both reasonably quick responses and
ratings such as urgency. This is a proposed next step based on our results,
not a claim that one model will win across all products.

### Laya's speed and local execution are worth preserving

At 24.1 ms, Laya's typical measured response was about five times faster than
Jev's and thirteen times faster than D1's. A faster decision could help a feature
respond promptly while a user is interacting with it. Those ratios compare this
run's recorded times; they are not measurements of maximum traffic capacity.

Laya was also the only model we ran locally. Its documentation confirms that it
runs on the user's own hardware. For a product that needs local processing,
that makes it a relevant candidate even though its initial answers were weaker.
[Source: Laya's documentation](https://github.com/NandhaKishorM/laya/blob/main/docs/index.md).

Local inference could let us process inputs without sending them to a hosted
model service. That is an architectural opportunity, not a privacy guarantee:
we would still need to check the application's downloads, logging, and other
network activity. The team would also own the work of providing hardware and
keeping the local installation running.

We should not assume that the speed difference comes entirely from the models.
The hosted measurements include a trip to an external service, while Laya's
measurements cover a local call. Model initialization and weight downloads
happened before the timed benchmark. The saved output does not establish which
processor performed Laya's inference, and we did not test repeated runs or
performance under heavy use. The benchmark repeats each request's time on its
question records, so requests with more questions have more influence on the
reported middle time.

### Being sure is useful only when it matches being right

The most important concern with Laya is that some wrong answers looked very
certain. Among its 11 answers assigned more than 90% probability, only seven
matched the expected answer—about 64%. The average stated probability in that
group was about 96%. Jev matched 20 of 21 answers in the same probability range,
and D1 matched all 25. These groups include choice, yes/no, and rating questions;
for ratings, the check uses the most likely rating level.

Source: the notebook's saved probability-bin tables. These figures describe the
probability assigned to the selected answer, rather than every field a provider
might call “confidence.”

This matters when deciding which cases a feature can handle automatically and
which should go to a person. A rule such as “act when the model is more than 90%
sure” would have let four incorrect Laya answers through in this test. Even Jev
had one mismatch in that range.

Research has shown that a model's stated certainty can differ from its actual
accuracy and that adjusting those probabilities can improve their agreement.
[Source: Guo and colleagues, *On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599).
The product implication is to choose any automatic-action rule from measured
results on our own examples, including the consequences of mistakes.

The notebook also showed a Laya warning about a saved setting used to adjust
probabilities. It said affected confidence estimates should be treated as
unreliable. We cannot tell from the aggregate output which tested answers were
affected, so the warning deserves investigation; it does not explain away the
observed mistakes.

## Could training Laya on our examples improve it?

Yes, that is a reasonable experiment. Fine-tuning means giving the model
additional training on examples of the decisions we want it to make. We could
teach it our support categories, our urgency definitions, and examples that
reflect our customers' language.

Laya's maintainers report that a model trained for a particular set of decision
tasks improved from 36.2% to 76.6% accuracy on their 2,000-decision benchmark.
They provide training examples and also warn that shipped models can be too
sure of their answers. These are maintainer-reported results on a different
benchmark; they do not predict the improvement we would get.
[Source: Laya's fine-tuning and confidence guidance](https://github.com/NandhaKishorM/laya#fine-tuning).

I would treat this as a limited experiment with a clear goal: improve one useful
product task while keeping the local response time attractive. Training the
model to answer better and adjusting how sure it says it is are separate jobs.
We should measure both on examples that were not used for training or adjustment.
A successful experiment would show fewer costly mistakes, credible certainty
estimates, and acceptable speed on the intended hardware.

## What this test has not answered yet

There are only 33 questions, including just six ratings. One additional choice
mistake changes accuracy by roughly eight percentage points. Several questions
share the same situation, so they are not 33 unrelated customer cases. Some
labels also express judgment—for example, what counts as urgent—rather than a
fact with only one possible answer.

All models returned usable answers, but a usable answer can still be wrong.
This run does not establish reliability over time, performance on unfamiliar
customer inputs, or behavior when many users arrive at once. The saved report
also does not record the exact Jev model version, the Laya weights selected for
each request, or the hardware used, which limits how precisely we can reproduce
it later.

We also cannot rank the models by cost. The notebook's cost column is blank
because prices were not supplied. Local inference has hardware and operating
costs; hosted inference has service costs. Neither “local” nor the D1 model name
`d1:free` supplies a complete cost comparison.

## A practical next step

I would propose a product trial with three parts:

1. **Compare D1 and Jev on a larger set of real examples.** Choose one workflow,
   agree on the expected answers, and count the mistakes that matter most to
   customers. Record model versions and repeat the timing measurements.
2. **Try a focused Laya improvement.** Train it on that workflow, check its
   certainty estimates, and test it on separate examples. Measure whether the
   better answers retain its local speed advantage.
3. **Measure the whole feature.** Track correct decisions, human review work,
   waiting time, and operating cost. Begin with suggestions people can check
   before expanding automatic actions.

The value of this bakeoff is that it turns model selection into concrete product
tradeoffs. D1 earned a closer look for answer quality, Jev for its balance of
quality and speed, and Laya for fast local processing with room to investigate
specialized training. We now have evidence to guide the next experiment, while
keeping the final decision tied to our customers' needs.
