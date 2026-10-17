# Prepared public corrections — publication pending

These are local corrections through 5A for historical v2.1 and batch-4B reporting. No push, merge, Hub/Space
upload, GitHub About edit or portfolio publication has been performed. Publish
from the reviewed commit IDs recorded in the consensus execution table, not
from an arbitrary working tree. See BATCH_5.md for corrected defects and the remaining 5B/5C acceptance gates.

| Destination | Prepared source / required material |
|---|---|
| GitHub repository | README.md, README.es.md, DELIVERABLE.md, COMO_FUNCIONA.md, historical labels, corrected scatter, docs/HISTORICAL_EVIDENCE.md and its evidence files |
| HF model card | model_card.md → README.md; include ATTRIBUTIONS.md and docs/HISTORICAL_EVIDENCE.md + evidence so relative links resolve |
| Demo Space | demo/app.py and demo/README.md; preserve its four existing example files; include ATTRIBUTIONS.md |
| API Space | api/README.md; include ATTRIBUTIONS.md; include current INPUT_CONTRACT.md and BATCH_5.md |
| GitHub About | Use the text below; keep the existing repository name |
| Portfolio | Use the neutral project description below, and link the negative agent result; personal contributions require Julian's own confirmed account |

Proposed model-repository About:

> 26M-parameter multitask transformer for DESI spectra: redshift and masked reconstruction; v2.1 fine-tunes v1; demo and REST API; historical evaluation limits documented.

Proposed Copilot About:

> Spectral-analysis tools, MCP and an experimental Claude agent. In 150 historical cases, model accuracy was 139/150 versus 119/150 for the agent.

Neutral portfolio description:

> DESI Spectra FM combines a 26M-parameter transformer with Python inference,
> Gradio and a REST API. Spectra Copilot adds spectral-line heuristics, FAISS
> neighbors, reference retrieval and MCP access. The stored 150-case comparison
> found that the agent reduced accuracy. Consolidation addresses reproducibility,
> non-independent historical validation and input contracts. In the three validation-selected batch-4B pairs, reconstruction reduced
> outliers and met the prespecified criterion; generalization to other seeds
> or checkpoints is unestablished. CNN won that practical comparison, which
> also changed training inputs. It was not compared with served v2.1.

Code licensing is MIT. Data retain DESI CC BY 4.0 conditions; carry the EDR
citation, modification notice and prescribed acknowledgment from ATTRIBUTIONS.md
into redistributed data, model-card materials and derived works.

Credential status (2026-10-09): revocation and consumer rotation not verified.
The owner must revoke the historical Anthropic key and the key shared in this
session, replace affected consumers securely, and report only completion date
and status. Do not add credentials, fragments, balances or storage paths here.

The old `docs/img/demo.png` is retained only as a historical file and removed from README navigation because it contains refuted claims. Do not publish it; a replacement must be a real capture of the corrected application (5C). No replacement capture was fabricated. Definitive wheels/versioning, clean install and remote publication remain 5C.
