#!/o0usr/bin/env python
# coding=utf-8
model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2']
victim_model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2']
victim_datasets = [('imagenet', '<dataset path>')]
test_output_path = '<where to save the generated adversarial examples>'
attack_book = '<which images to attack>'

test_evaluation_file = 'evaluation/test_evaluation.csv'
attack_methods = {
    'IDAA': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 1.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          # decay factor
    },
}
