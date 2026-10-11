"""Deterministic synthetic ticket teacher; no external labeling service."""
import json
import random
from collections import Counter
from pathlib import Path

from system_one_models.finetuning import file_digest, verify_dataset

OPTIONS = {'billing': 'payment and charge issues', 'sales': 'new purchases', 'support': 'general help'}
QUESTION = 'Which team handles this?'
# Families, rather than near-duplicate rows, are assigned to splits.
TEMPLATES = {
    'billing': [
        'I was charged twice for order {number}. Please reverse the duplicate payment.',
        'My invoice {number} has an incorrect tax amount. Can you correct the charge?',
        'The receipt for order {number} shows a payment I did not authorize.',
        'Please refund the cancelled subscription on account {number}.',
        'My card payment for invoice {number} failed. Help me pay the outstanding balance.',
        'You billed account {number} after cancellation. Please return the money.',
        'I need a refund for an unused annual subscription, reference {number}.',
        'There is an unexpected fee on statement {number}. Explain this charge.',
        'The bank transfer for invoice {number} has not been credited to my account.',
        'Our purchase {number} was paid but the bill still says overdue.',
        'Please update the payment card for subscription {number}.',
        'Where can I get a corrected receipt for transaction {number}?',
        'Our statement lists two payments for reference {number}; only one is valid.',
        'Can you investigate the missing reimbursement for return {number}?',
        'A charge appeared after my free trial ended, account {number}. I want it refunded.',
        'The amount withdrawn for purchase {number} differs from the agreed invoice.',
        'Our accounts payable team needs help reconciling invoice {number}.',
        'Please remove the late-payment penalty from bill {number}.',
        'The amount due on statement {number} does not include our credit.',
        'We sent money for purchase {number}; your ledger still shows an unpaid balance.',
    ],
    'sales': [
        'We want to buy {number} seats. Please send enterprise pricing.',
        'Can we schedule a product demonstration for a new purchase of {number} licenses?',
        'Please quote a new annual agreement for {number} users.',
        'We are comparing vendors for {number} staff. What packages do you offer?',
        'Our procurement team is considering {number} seats. Can we discuss a contract?',
        'I want to purchase {number} subscriptions. Is there a volume discount?',
        'Can someone walk us through pricing before we order {number} accounts?',
        'Please share a proposal for a new deployment to {number} employees.',
        'We plan to buy {number} licenses next quarter. Can you provide a quote?',
        'Which plan should we purchase for a team of {number}?',
        'We need commercial terms for {number} new subscriptions.',
        'How much would a new contract for {number} seats cost?',
        'Please arrange a presales consultation for {number} prospective users.',
        'We have approval to purchase {number} seats. Who can finalize our order?',
        'I would like a trial demonstration before buying {number} licenses.',
        'We are shopping for a platform for {number} staff. Please explain your offerings.',
        'Our budget covers {number} new seats. Send a purchasing proposal.',
        'Are there discounts for a first-time order of {number} subscriptions?',
        'Please put together an offer for {number} new accounts at our company.',
        'We have not used your service yet. Can we discuss buying access for {number} people?',
    ],
    'support': [
        'My password reset link expired, ticket {number}. Please help me sign in.',
        'The dashboard crashes when I open settings, case {number}.',
        'I cannot export my report, reference {number}. How can I fix this?',
        'The application shows a blank screen, case {number}. Please troubleshoot.',
        'Help me configure notifications on account {number}.',
        'How do I add a colleague to my existing workspace {number}?',
        'My uploaded file will not open, ticket {number}.',
        'The search function returns an error, reference {number}.',
        'I forgot my login password for account {number}.',
        'How can I change the timezone in workspace {number}?',
        'The mobile app freezes on launch, case {number}.',
        'Please help restore a deleted report, reference {number}.',
        'Where can I find the export button in project {number}?',
        'The API returns an authentication error for workspace {number}.',
        'My existing integration stopped syncing, ticket {number}.',
        'I need instructions for enabling dark mode on account {number}.',
        'A colleague cannot access our shared folder, reference {number}.',
        'How do I rename project {number} in the settings menu?',
        'Every time I try to log in, the page reloads, case {number}.',
        'Our saved charts disappeared from workspace {number}. Can you recover them?',
    ],
}


def generate(directory: str | Path, count: int = 3000, seed: int = 42) -> None:
    """Freeze an 80/10/10 split, dividing validation equally for stopping/calibration."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Use a new empty directory; frozen datasets are never overwritten')
    if count < 120 or count % 60:
        raise ValueError('Count must be a multiple of 60 and at least 120')
    rng = random.Random(seed)
    splits = {name: [] for name in ('train', 'validation', 'calibration', 'test')}
    for label, templates in TEMPLATES.items():
        order = list(range(20))
        rng.shuffle(order)
        for position, index in enumerate(order):
            split = ('train' if position < 16 else 'validation' if position == 16
                     else 'calibration' if position == 17 else 'test')
            for variant in range(count // 60):
                # Reference values are nuisance variation, never the teacher's decision rule.
                number = rng.randrange(100, 99999)
                state = templates[index].format(number=number)
                state += f' Request reference: {rng.getrandbits(64):016x}.'
                row = {'id': f'{seed}-{label}-{index}-{variant}', 'family': f'{label}-{index}',
                       'state': state, 'question': QUESTION, 'options': OPTIONS, 'label': label}
                splits[split].append(row)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = {'seed': seed, 'examples': count, 'teacher': 'single-intent template family',
                'split_policy': '16/1/1/2 disjoint families per class',
                'sha256': {s: file_digest(directory / f'{s}.jsonl') for s in splits},
                'counts': {s: dict(Counter(r['label'] for r in rows)) for s, rows in splits.items()}}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Spot-check only training/validation; do not expose the frozen test labels in the notebook.
    sample = rng.sample(splits['train'], min(100, len(splits['train'])))
    (directory / 'spot-check.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in sample))
    verify_dataset(directory)
