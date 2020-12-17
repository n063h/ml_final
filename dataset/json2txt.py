import os,json

def get_obj(filename):
    with open(filename,'r') as f:
        j = json.load(f)
    bbox=j['bbox']
    return j['flaw_type'],list(bbox.values())

def read(data_root):
    label_root=os.path.join(data_root,'label_json')
    temp_root=os.path.join(data_root,'temp')
    trgt_root = os.path.join(data_root, 'trgt')
    time_dev_dirs = os.listdir(label_root)
    obj=[]
    cnt=0
    for time_dev_dir in time_dev_dirs:
        label_dir = os.path.join(label_root, time_dev_dir)
        temp_dir = os.path.join(temp_root, time_dev_dir)
        trgt_dir = os.path.join(trgt_root, time_dev_dir)
        label_jsons = os.listdir(label_dir)
        for json_file in label_jsons:
            file_name=json_file.split('.')[0]
            temp_path = os.path.join(temp_dir, file_name+'.jpg')
            trgt_path = os.path.join(trgt_dir, file_name + '.jpg')
            label_path=os.path.join(label_dir, file_name + '.json')
            cls,img_obj=get_obj(label_path)
            obj.append({'temp_path':temp_path,'trgt_path':trgt_path,'img_obj':img_obj,'cls':cls})
            cnt += 1
            if cnt % 100 == 0:
                print('read complete %d', cnt)
    return obj

def write(txt_path,data):
    with open(txt_path,'w') as f:
        for d in data:
            if d['cls']>14:continue
            obj_box=d['img_obj']
            #temp_path trgt_path cls x0 x1 y0 y1
            f.write('%s %s %s %s+%s+%s+%s\n'%(d['temp_path'],d['trgt_path'],str(d['cls']),str(obj_box[0]),str(obj_box[1]),str(obj_box[2]),str(obj_box[3])))

if __name__ == '__main__':
    data=read('./dataset/fabric_data_new')
    write('./dataset/all_data.txt',data)
    write('./dataset/train_data.txt', data[:1200])
    write('./dataset/test_data.txt', data[1200:])