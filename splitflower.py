import os
import shutil
import random

def splitdata(src_path,dest_train_path,dest_test_path):
    splitpath = [[]for i in range(17)]
    ncount = 0
    for image_name in os.listdir(src_path):
        img_path = os.path.join(src_path,image_name)
        print(ncount//80)
        splitpath[ncount//80].append(img_path)
        ncount += 1

    for i in range(17):
        train_dest_dir = os.path.join(dest_train_path,str(i+1))
        test_dest_dir = os.path.join(dest_test_path,str(i+1))
        os.mkdir(train_dest_dir)
        os.mkdir(test_dest_dir)
        nmark = 0
        random.shuffle(splitpath[i])
        for img_path in splitpath[i]:
            if(nmark//(len(splitpath[i])/5)<1):
                shutil.copy(img_path,test_dest_dir)
            else:
                shutil.copy(img_path,train_dest_dir)
            nmark +=1


if __name__ == '__main__':
    splitdata("17flowers/jpg","data/train_flower","data/test_flower")


