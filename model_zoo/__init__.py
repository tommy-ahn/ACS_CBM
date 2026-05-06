from .cbm import ConceptBottleneckModel
from .cem import ConceptEmbeddingModel
from .probcbm import ProbabilisticConceptBottleneckModel
from .ecbm import EnergyConceptBottleneckModel
from .ecbm2 import EnergyConceptBottleneckModelv2
from .grace import GraceConceptBottleneckModel


def create_model(cfg, imbalance):
    model = None
    if cfg.MODEL.NAME.lower() == "cbm":
        model = ConceptBottleneckModel(cfg, imbalance)
    elif cfg.MODEL.NAME.lower() == "cem":
        model = ConceptEmbeddingModel(cfg, imbalance)
    elif cfg.MODEL.NAME.lower() == "probcbm":
        model = ProbabilisticConceptBottleneckModel(cfg, imbalance)
    elif cfg.MODEL.NAME.lower() == "ecbm":
        model = EnergyConceptBottleneckModel(cfg, imbalance)
    elif cfg.MODEL.NAME.lower() == "ecbm2":
        model = EnergyConceptBottleneckModelv2(cfg, imbalance)
    elif cfg.MODEL.NAME.lower() == "grace_cbm":
        model = GraceConceptBottleneckModel(cfg, imbalance)
    else:
        raise NotImplementedError(f"model : {cfg.MODEL.NAME} is not implemented")

    # model = LitModelWrapper(cfg, model, imbalance)
    return model
