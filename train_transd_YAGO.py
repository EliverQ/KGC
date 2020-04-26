import my_oke

from my_oke.module.model import TransD
from my_oke.module.loss_fun import MarginLoss
from my_oke.module.SamplingStrategy import NegativeSampling

from my_oke.config import Trainer, Tester

from my_oke.dataloader import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./datasets/YAGO3-10/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./datasets/YAGO3-10/", "link")

# define the model
transd = TransD(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 200, 
	dim_r = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transd, 
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transd.save_checkpoint('./checkpoint/transd.ckpt')

# test the model
transd.load_checkpoint('./checkpoint/transd.ckpt')
tester = Tester(model = transd, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
