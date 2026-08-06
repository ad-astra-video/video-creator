"""Local in-process ID-V2V restyle pipeline (compatible-GPU gate).

The desktop can serve Restyle through its own GPU when a compatible card is
present, mirroring the existing local LTX path. The portable id-v2v engine core
(``runner/idv2v/model.py`` + ``run.py``) has no HTTP dependency, so it is driven
in-process exactly like an LTX local pipeline.

Importing the id-v2v model stack (diffsynth fork + torchao) pulls heavy deps
that are NOT in the desktop backend venv by default — they ship with the
idv2v_worker container. So this service lazily imports ``runner.idv2v`` only
when actually used and the deps are present; any ImportError degrades to remote
fallback (live-runner -> idv2v-worker), keeping the desktop usable everywhere.
"""


from .local_idv2v_pipeline import LocalIdV2vPipeline, LocalIdV2vPipelineUnavailable

__all__ = ["LocalIdV2vPipeline", "LocalIdV2vPipelineUnavailable"]
