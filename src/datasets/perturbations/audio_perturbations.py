import src.utils as utils
import torch

class AudioPerturbations():
    def __init__(self, perturbations_list) -> None:
        
        self.perturbations = []
        self.perturbation_prob = []
        
        for perturbation_desc in perturbations_list:
            assert 'type' in perturbation_desc, 'Perturbation has no specified type!'
            assert 'prob' in perturbation_desc, 'Perturbation has no specified probability!'

            # If no params are specified, assume there are no params given
            if 'params' not in perturbation_desc:
                perturbation_desc['params'] = {}

            perturbation = utils.import_attr(perturbation_desc['type'])(**perturbation_desc['params'])
            self.perturbations.append(perturbation)
            self.perturbation_prob.append(perturbation_desc['prob'])

    def apply_random_perturbations(self, input_audio, gt_audio):
        perturbed_input_audio = input_audio
        perturbed_gt_audio = gt_audio

        # Go over perturbations
        for prob, perturbation in zip(self.perturbation_prob, self.perturbations):
            # With some probability, apply this perturbation
            if torch.rand((1,)).item() < prob:
                perturbed_input_audio, perturbed_gt_audio = \
                    perturbation(perturbed_input_audio, perturbed_gt_audio)
        
        return perturbed_input_audio, perturbed_gt_audio
