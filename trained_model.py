from models.modules import ERFNetModel
from models.modules import AleatoricERFNetModel
import torch
from utils.utils import *


class SemanticSegmenter:
    def __init__(self, ckpt_path):
        ckpt_file = torch.load(ckpt_path)
        cfg = ckpt_file["hyper_parameters"]
        model_weights = ckpt_file["state_dict"]

        for key in list(model_weights):
            model_weights[key.replace("model.", "")] = model_weights.pop(key)

        if cfg["model"]["name"] == "erfnet":
            model = ERFNetModel
        elif cfg["model"]["name"] == "erfnet_w_aleatoric":
            model = AleatoricERFNetModel

        self.model = model(**cfg)
        self.model.load_state_dict(model_weights)

        self.num_mc_dropout = cfg["num_mc_dropout"]
        self.aleatoric_model = cfg["aleatoric_model"]
        self.num_mc_aleatoric: cfg["num_mc_aleatoric"]

        self.use_mc_dropout = self.num_mc_dropout > 1
        self.num_mc_dropout = self.num_mc_dropout if self.num_mc_dropout > 1 else 1

        self.num_predictions = self.num_mc_dropout
        self.device = "cuda"
        self.softmax = nn.Softmax(dim=1)

    def predict(self, image):
        image = torch.tensor(image).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        self.predictions = []
        self.hidden_representations = []

        single_model = self.model.to(self.device)
        single_model.eval()
        if self.use_mc_dropout:
            enable_dropout(single_model)

        for i in range(self.num_predictions):
            with torch.no_grad():
                if self.aleatoric_model:
                    est_anno, hidden_representation = sample_from_aleatoric_model(
                        single_model,
                        image,
                        num_mc_aleatoric=self.num_mc_aleatoric,
                        device=self.device,
                    )
                else:
                    est_anno, hidden_representation = single_model.forward(image)

                est_seg_probs = self.softmax(est_anno)
                self.predictions.append(est_seg_probs.cpu().numpy())
                self.hidden_representations.append(hidden_representation.cpu().numpy())

        (
            mean_predictions,
            variance_predictions,
            entropy_predictions,
            mutual_info_predictions,
            hidden_representations,
        ) = compute_prediction_stats(
            np.array(self.predictions), np.array(self.hidden_representations)
        )

        uncertainty_predictions = (
            mutual_info_predictions if self.use_mc_dropout else entropy_predictions
        )

        return mean_predictions, uncertainty_predictions, hidden_representations


def get_trained_segmenter(ckpt_path):
    return SemanticSegmenter(ckpt_path)
