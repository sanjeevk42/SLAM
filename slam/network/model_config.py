import json


class ModelConfigProvider:
    
    def __init__(self):
        with open('resources/model.config') as f:
            contents = f.read()
        self.config = json.loads(contents, encoding='utf-8')
    
    def training_filenames(self):
        return self.config['train']['slam_files']
    
    def test_filenames(self):
        return self.config['test']['slam_files']
    
    def lstm_layers(self):
        return self.config['train']['model']['lstm_layers']
    
    def cnn_output_dim(self):
        return self.config['train']['model']['cnn_output_dim']
    
    def epoch(self):
        return self.config['train']['model']['epoch']
    
    def sequence_length(self):
        return self.config['train']['model']['sequence_length']
    
    def batch_size(self):
        return self.config['train']['model']['batch_size']
        
    def learning_rate(self):
        return self.config['train']['model']['learning_rate']

    def  normalization_epsilon(self):
        return self.config['train']['model']['normalization_epsilon']


config_provider = ModelConfigProvider()

def get_config_provider():
    return config_provider
    
