import json


class ModelConfigProvider:
    
    def __init__(self):
        with open('../resources/model.config') as f:
            contents = f.read()
        self.config = json.loads(contents, encoding='utf-8')
    
    def get_training_filenames(self):
        return self.config['train']['slam_files']
    
    def get_test_filenames(self):
        return self.config['test']['slam_files']
    
    def get_lstm_layers(self):
        return self.config['train']['model']['lstm_layers']
    
    

config_provider = ModelConfigProvider()

def get_config_provider():
    return config_provider
    