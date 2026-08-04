from .node_base import DataType

# Compatibility matrix: source_type -> set of compatible target types
_COMPAT: dict[DataType, set[DataType]] = {
    DataType.TENSOR: {DataType.TENSOR, DataType.ANY},
    DataType.MODEL: {DataType.MODEL, DataType.ANY},
    DataType.DATASET: {DataType.DATASET, DataType.ANY},
    DataType.DATALOADER: {DataType.DATALOADER, DataType.ANY},
    DataType.OPTIMIZER: {DataType.OPTIMIZER, DataType.ANY},
    DataType.LOSS_FN: {DataType.LOSS_FN, DataType.ANY},
    DataType.SCALAR: {DataType.SCALAR, DataType.ANY},
    DataType.STRING: {DataType.STRING, DataType.ANY},
    DataType.IMAGE: {DataType.IMAGE, DataType.TENSOR, DataType.ANY},
    DataType.LIST: {DataType.LIST, DataType.ANY},
    DataType.TRANSFORM: {DataType.TRANSFORM, DataType.ANY},
    DataType.ANY: {dt for dt in DataType if dt is not DataType.TRIGGER},
    DataType.TRIGGER: {DataType.TRIGGER},
}


def is_compatible(source: DataType, target: DataType) -> bool:
    if source == DataType.TRIGGER or target == DataType.TRIGGER:
        return source == DataType.TRIGGER and target == DataType.TRIGGER
    if target == DataType.ANY:
        return True
    return target in _COMPAT.get(source, set())
