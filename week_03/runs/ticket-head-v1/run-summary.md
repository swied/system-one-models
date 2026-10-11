# Ticket-routing head run

Completed real head-only training and calibration on the 3,000-example synthetic
dataset. The 300-example frozen test stayed at 83.3% accuracy; ECE improved from
0.231 (base) to 0.074 (trained and calibrated). Full tuning exceeded the observed
RTX 2080 GPU's approximately 8 GB capacity.

See the [reviewed results](../../docs/head-only-results.md) for measurements,
provenance, interpretation, and limitations. Generated weights/reports are ignored.
