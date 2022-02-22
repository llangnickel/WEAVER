from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForTokenClassification, AutoConfig
from transformers import HfArgumentParser
import transformers
import torch
import logging
import os
from collections import OrderedDict
from tqdm import tqdm
os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ModelArguments:
    """
    Arguments for the models that should be merged and the amounts of training data sets for weighted averaging.
    """
    output_dir: str = field(
        metadata={"help": "Path where the new model should be saved"}
    )
    baseline_model: str = field(
        metadata={"help": "Name or path of pre-trained transformer model."}
    )
    amount_of_labels: int = field(
        metadata={"help": "Number of labels the models were trained on."}
    )
    first_model_path: str = field(
        metadata={"help": "Path to the first model."}
    )
    amount_first_model: int = field(
        metadata={"help": "Amount of training data used for training the model."}
    )
    second_model_path: str = field(
        metadata={"help": "Path to the second model."}
    )
    amount_second_model: int = field(
        metadata={"help": "Amount of training data used for training the model."}
    )
    further_models: Optional[str] = field(
        default="", metadata={"help": "Whitespace-separated list containing further models that should be merged for"
                                      "federated learning settings."}
    )
    further_amounts: Optional[str] = field(
        default="", metadata={"help": "Whitespace-separated list consisting of further amounts corresponding to the"
                                      "amount of training data the further models were trained on"}
    )


def average_weights(input_models, coefficients):
    """average weights of different transformer models based on the amount of training data they were trained on"""
    weights_averaged = OrderedDict()
    for i, current_model in tqdm(enumerate(input_models), leave=False):
        current_weights = current_model.state_dict()
        for key in current_weights.keys():
            if i == 0:
                weights_averaged[key] = coefficients[i] * current_weights[key]
            else:
                weights_averaged[key] += coefficients[i] * current_weights[key]

    return weights_averaged


def update_a_model(base_model, weights):
    """update a base model with new weights"""

    base_model.load_state_dict(weights)

    return base_model


if __name__ == '__main__':
    transformers.logging.set_verbosity_error()
    logger = logging.getLogger(__name__)
    # parse arguments
    parser = HfArgumentParser(ModelArguments)
    args = parser.parse_args()

    # baseline_model = "bert-base-cased"
    models_to_merge = []
    training_data_amounts_temp = []

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.baseline_model,
        num_labels=args.amount_of_labels
    )

    # load models
    # baseline
    baseline_model = AutoModelForTokenClassification.from_pretrained(
        args.baseline_model,
        from_tf=bool(".ckpt" in args.baseline_model),
        config=config
    )

    first_model_path = args.first_model_path
    first_model = AutoModelForTokenClassification.from_pretrained(
        first_model_path,
        from_tf=bool(".ckpt" in first_model_path),
        config=config
    )
    models_to_merge.append(first_model)
    training_data_amounts_temp.append(args.amount_first_model)

    second_model_path = args.second_model_path
    second_model = AutoModelForTokenClassification.from_pretrained(
        second_model_path,
        from_tf=bool(".ckpt" in second_model_path),
        config=config
    )
    models_to_merge.append(second_model)
    training_data_amounts_temp.append(args.amount_second_model)

    further_models = args.further_models.split()
    further_amounts = args.further_amounts.split()
    assert len(further_models) == len(further_amounts), "For each trained model a corresponding amount of the used " \
                                                        "training data is needed "
    for further_model, amount in zip(further_models, further_amounts):
        model = AutoModelForTokenClassification.from_pretrained(
            further_model,
            from_tf=bool(".ckpt" in further_model),
            config=config
        )
        models_to_merge.append(model)
        training_data_amounts_temp.append(int(amount))

    total_amount = sum(training_data_amounts_temp)
    training_data_amounts = [c / total_amount for c in training_data_amounts_temp]

    averaged_weights = average_weights(models_to_merge, training_data_amounts)
    updated_model = update_a_model(baseline_model, averaged_weights)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    torch.save(updated_model.state_dict(), output_model_file)
    CONFIG_NAME = "config.json"
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(updated_model.config.to_json_string())
    print('Model was saved to ', args.output_dir)
    with open(os.path.join(args.output_dir, "merged_models.txt"), "w") as f:
        print("--------------------------------------", file=f)
        print(f"Model 1: {first_model_path}", file=f)
        print(f"Model 2: {second_model_path}", file=f)
        c = 3
        for model_path in args.further_models.split():
            print(f"Model {str(c)}: {model_path}", file=f)
            c += 1
        print(f"Coefficients: {str(training_data_amounts)}", file=f)
