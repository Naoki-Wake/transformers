from azure.storage.blob import BlobServiceClient
import os.path as osp
import os
import time
start = time.process_time()
service = BlobServiceClient(account_url="URL", 
   credential={"account_name": "NAME", "account_key":"KEY"})
container_client = service.get_container_client(container="sthv2")
# List blobs in container
iter_blob = container_client.list_blobs("dataset/features/videoMAE/")

#start = time.process_time()
#list_blob = list(iter_blob)
#print(time.process_time() - start)
print("listdir")

calculated_list = os.listdir('/home/nawake/ssv2/dataset/features/videoMAE/')
print(time.process_time() - start)
print(len(calculated_list))
#import time
#start = time.process_time()
#sum(1 for _ in iter_blob)
#print(time.process_time() - start)
#for item in iter_blob:
#    print(osp.basename(item.name))
#    import pdb;pdb.set_trace()