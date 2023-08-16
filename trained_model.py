from models.modules import ERFNetModel
from models.modules import AleatoricERFNetModel
import torch
from utils.utils import *


class SemanticSegmenter:
    def __init__(self, ckpt_path, uncertainty_mode, device):
        self.device = device
        self.uncertainty_mode = uncertainty_mode
        ckpt_file = torch.load(ckpt_path)
        cfg = ckpt_file["hyper_parameters"]["cfg"]
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
        self.softmax = nn.Softmax(dim=1)

    def predict(self, image):
        image = torch.tensor(image).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        prob_predictions = []
        aleatoric_unc_predictions = []

        self.single_model = self.model.to(self.device)
        self.single_model.eval()
        if self.use_mc_dropout:
            enable_dropout(self.single_model)

        for i in range(self.num_predictions):
            with torch.no_grad():
                if self.aleatoric_model:
                    est_prob, est_aleatoric_unc = self.sample_aleatoric(image)
                else:
                    est_prob, _ = self.single_model(image)
                    est_prob, est_aleatoric_unc = self.softmax(
                        est_prob
                    ), torch.zeros_like(est_prob[:, 0, :, :], device=self.device)

                prob_predictions.append(est_prob)
                aleatoric_unc_predictions.append(est_aleatoric_unc.squeeze(1))

        prob_predictions = torch.stack(prob_predictions)
        aleatoric_unc_predictions = torch.stack(aleatoric_unc_predictions)

        (
            mean_predictions,
            epistemic_variance_predictions,
            predictive_entropy_predictions,
            epistemic_mutual_info_predictions,
        ) = compute_prediction_stats(prob_predictions)
        aleatoric_unc_predictions = torch.mean(aleatoric_unc_predictions, dim=0)

        if self.uncertainty_mode == "predictive":
            uncertainty_predictions = predictive_entropy_predictions
        elif self.uncertainty_mode == "epstemic":
            uncertainty_predictions = epistemic_mutual_info_predictions
        elif self.uncertainty_mode == "aleatoric":
            uncertainty_predictions = aleatoric_unc_predictions
        elif self.uncertainty_mode == "comnbined":
            uncertainty_predictions = (
                epistemic_mutual_info_predictions + aleatoric_unc_predictions
            )
        else:
            raise RuntimeError("unknown uncertainty mode")

        return mean_predictions, uncertainty_predictions

    def sample_aleatoric(self, image):
        est_seg, est_std, _ = self.single_model(image)
        sampled_predictions = torch.zeros(
            (self.num_mc_aleatoric, *est_seg.size()), device=self.device
        )
        for j in range(self.num_mc_aleatoric):
            noise_mean = torch.zeros(est_seg.size(), device=self.device)
            noise_std = torch.ones(est_seg.size(), device=self.device)
            epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
            sampled_seg = est_seg + torch.mul(est_std, epsilon)
            sampled_predictions[j] = self.softmax(sampled_seg)

        mean_predictions, _, entropy_predictions, _ = compute_prediction_stats(
            sampled_predictions
        )
        return (
            mean_predictions,
            entropy_predictions,
        )


def get_segmenter(args, device):
    ckpt_path = args.ckpt_path
    uncertainty_mode = args.uncertainty_mode
    return SemanticSegmenter(ckpt_path, uncertainty_mode, device)
