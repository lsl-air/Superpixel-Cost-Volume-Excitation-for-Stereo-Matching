# from models.gwcnet import GwcNet_G, GwcNet_GC
from models.gwcnet import GwcNet_G, GwcNet_GC
from models.loss import model_loss,compute_slic_loss,cost_loss,test_model_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-gc": GwcNet_GC
}
