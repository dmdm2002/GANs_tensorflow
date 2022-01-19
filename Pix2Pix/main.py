from Trainner import Trainner

starter = Trainner()
run_type = 'train'

if run_type == 'train':
    starter.training()
else:
    starter.testing()