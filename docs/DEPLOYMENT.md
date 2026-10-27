# Hosted applications and package releases

This software release does not update the code of the hosted demo or API.
Their FM dependency currently points to GitHub's default branch without a
version pin. The installed package version is not attested here.

A future Space rebuild can resolve a different FM package even when this
publication does not touch that Space. A README commit can trigger a rebuild
and restart; caching may retain a dependency and builds can fail. Reverting a
README does not restore a previous runtime. An HTTP availability check does not
identify the installed package and may wake a sleeping application.

Updating the hosted applications is separate work: pin the dependency to
`v0.3.0`, align application text with the documented scientific limits and test
the application locally before deploying. No updated hosted demo is claimed.
The historical examples are not blind evidence of generalization, and the
repository name does not establish foundation-model capability.

Use [REPRODUCE.md](REPRODUCE.md) for a local run with fixed public assets and
[EVALUATION.md](EVALUATION.md) for evidence and limitations.
