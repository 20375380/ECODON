# for basic training plan
import torch

class BasicTrainer():
    def __init__(self, train_config, Trainloader, Valloader, device, model, criterion, optimizer,
                 save_path, lr):
        self.config = train_config
        self.trainloader = Trainloader
        self.valloader = Valloader
        self.device = device
        self.model = model
        self.criterion=criterion
        self.optimizer = optimizer(model.parameters(), lr = lr)
        self.save_path = save_path

    def training(self, start_time):
        self.model.to(self.device)
        model_name = self.save_path + f"model_{start_time}.pth"
        model_weight = self.save_path +f"modeWeight_{start_time}.pth"
        model_final = self.save_path + f"finalModel_{start_time}.path"
        epoch_train_loss=[]
        epoch_val_loss=[]
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss=[]
            for one_batch in self.trainloader:
                amino_seq, struct_seq, gene_seq = [ele.to(self.device) for ele in one_batch]
                pred_gene = self.model(amino_seq, struct_seq)
                loss = self.criterion(pred_gene, gene_seq)
                train_loss.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            ave_train_loss = sum(train_loss)/len(train_loss)
            epoch_train_loss.append(ave_train_loss)

            self.model.eval()
            val_loss = []
            for one_batch in self.valloader:
                amino_seq, struct_seq, gene_seq = [ele.to(self.device) for ele in one_batch]
                pred_gene = self.model(amino_seq, struct_seq)
                val_loss.append(self.criterion(pred_gene, gene_seq))

            ave_val_loss = sum(loss)/len(loss)
            epoch_val_loss.append(ave_val_loss)

            if (epoch+1)%10 == 0:
                print(f"Epoch:{epoch+1}\t train loss:{ave_train_loss:6f}\t val loss:"
                      f"{ave_val_loss:6f}")

            if (epoch+1)%50 == 0:
                torch.save(self.model.state_dict(), model_weight)
                torch.save(self.model,model_name)






        torch.save(self.model, model_final)








