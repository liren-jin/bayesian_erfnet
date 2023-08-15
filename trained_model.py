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

        model_name = cfg["model"]["name"]
        num_classes = cfg["model"]["num_classes"]
        epistemic_version = (
            "standard"
            if "epistemic_version" not in cfg["model"]
            else cfg["model"]["epistemic_version"]
        )

        in_channels = cfg["model"]["in_channels"]
        dropout_prob = cfg["model"]["dropout_prob"]
        deep_encoder = cfg["model"]["deep_encoder"]
        shared_decoder = cfg["model"]["shared_decoder"]

        self.num_mc_aleatoric = cfg["train"]["num_mc_aleatoric"]
        self.num_mc_epistemic = cfg["train"]["num_mc_epistemic"]

        if model_name == "erfnet":
            self.model = ERFNetModel(
                num_classes,
                in_channels,
                dropout_prop=dropout_prob,
                deep_encoder=deep_encoder,
                epistemic_version=epistemic_version,
            )
            self.aleatoric_model = False
        elif model_name == "erfnet_w_aleatoric":
            self.model = AleatoricERFNetModel(
                num_classes,
                in_channels,
                dropout_prop=dropout_prob,
                use_shared_decoder=shared_decoder,
                deep_encoder=deep_encoder,
                epistemic_version=epistemic_version,
            )
            self.aleatoric_model = True

        self.model.load_state_dict(model_weights)

        self.use_mc_dropout = self.num_mc_dropout > 1
        self.num_mc_dropout = self.num_mc_dropout if self.num_mc_dropout > 1 else 1

        self.num_predictions = self.num_mc_dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.softmax = nn.Softmax(dim=1)

    def predict(self, image):
        image = torch.tensor(image).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        self.predictions = []
        self.hidden_representations = []

        self.single_model = self.model.to(self.device)
        self.single_model.eval()
        if self.use_mc_dropout:
            enable_dropout(self.single_model)

        for i in range(self.num_predictions):
            with torch.no_grad():
                if self.aleatoric_model:
                    est_anno, hidden_representation = self.sample_aleatoric(image)
                else:
                    est_anno, hidden_representation = self.single_model(image)

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

    def sample_aleatoric(self, image):
        est_seg, est_std, hidden_representation = self.single_model(image)
        sampled_predictions = torch.zeros(
            (self.num_mc_aleatoric, *est_seg.size()), device=self.device
        )
        for j in range(self.num_mc_aleatoric):
            noise_mean = torch.zeros(est_seg.size(), device=self.device)
            noise_std = torch.ones(est_seg.size(), device=self.device)
            epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
            sampled_seg = est_seg + torch.mul(est_std, epsilon)
            sampled_predictions[j] = sampled_seg
        return torch.mean(sampled_predictions, dim=0), hidden_representation


def get_trained_segmenter(ckpt_path):
    return SemanticSegmenter(ckpt_path)
