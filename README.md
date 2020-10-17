<!-- <div dir="rtl"><a href="https://github.com/ellerbrock/open-source-badges/"><img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" alt="Open Source Love"></a><a href="https://firstcontributions.herokuapp.com">[<img align="left" width="150" src="../assets/join-slack-team.png">](https://join.slack.com/t/firstcontributors/shared_invite/enQtNjkxNzQwNzA2MTMwLTVhMWJjNjg2ODRlNWZhNjIzYjgwNDIyZWYwZjhjYTQ4OTBjMWM0MmFhZDUxNzBiYzczMGNiYzcxNjkzZDZlMDM)</a></div> -->

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[<img align="right" width="150" src="../assets/join-slack-team.png">](https://join.slack.com/t/firstcontributors/shared_invite/enQtMzE1MTYwNzI3ODQ0LTZiMDA2OGI2NTYyNjM1MTFiNTc4YTRhZTg4OWZjMzA0ZWZmY2UxYzVkMzI1ZmVmOWI4ODdkZWQwNTM2NDVmNjY)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open Source Helpers](https://www.codetriage.com/roshanjossey/first-contributions/badges/users.svg)](https://www.codetriage.com/roshanjossey/first-contributions)


# <div dir="rtl">المقدمة </div>

<div dir="rtl">
يعتبر الفيدو من المصادر المهمة للمعلومات ولبناء انظمة مختلفة    ونحن هنا سنستخدم الفيديو للكشف عن حالات العنف سواء كان فيديو من كامرات مراقبة او فيديو  في مقطع فلم او عبر الانترنت<br>

كبداية  هنالك ثلاثة طرق  شائعه للتعامل مع كشف  الحركة او معالجة الفيديو بالتعلم العميق
وهذه الطرق
<br>
(Conv3d) 
<br>
(Convlstm) 

<br>
(Conv  +  lstm) :   هذه الطريقة تتيح لنا استخدام  شبكة كونفليوشن مدربة مسبقا  لتحسين النتائج وهي الطريقة الاكثر شيوعا عندما يكون  لدينا عدد بيانات قليل وايضا قدرة الحاسوب من ناحية كارت الشاشة قليلة  ولذلك سنقوم بعمل الشرح على هذه الطريقة وايضا لن هذه الطريقة تحتج جهد خاص لتطويعها وفق بيئة  pytorch



<br>  ماذا ستتعلم من هذا المقال

<br> 1-  بناء كلاس  خاص لقراءة المحتوى الفيديوي  للpytorch

<br> 2-  بناء  كلاس خاص  لتوزيع الصور من الفيديو الواحد على الكونفليوشن الاعتيادية    مشابة  لل  time-distrbution in keras

<br> 3-  استخدمات  الlstm   بسهولة بداخل   ال  sequnatil mode in pytorch
</div>

# <div dir="rtl"> المتطلبات الاساسية </div>

<div dir="rtl">


<br> 1-  ان تكون بمستوى متوسط  بالباي تورش   اي ان تستطيع على الاقل انشاء مودل وتدريبة  

<br> 2-  تكون المكتبات التالية مثبته لديك   pytorch , opencv , pandas

<br> 3-  البيانات المستخدمة بالتدريب يمكنك تحميلها وايجادها من الروابط التالية علما انني قمت بدمجها ثلاثتها وتدريب المودل عليها
 
 (Movies Fight Detection Dataset https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635 ) , (Hockey Fight Detection Dataset https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89 ) , and (VIOLENT-FLOWS DATABASE https://www.openu.ac.il/home/hassner/data/violentflows/ )

</div>



## <div dir="rtl">خطوات العمل  </div>

## <div dir="rtl">1- بناء  كلاس قراءة وتحويل الفيديو </div>

<div dir="rtl">يوفر لنا باي تورش  سهولة بالتعامل وتوسيع الكلاسات الخاصة به   واهمها كلاس      Dataset  ك معالينا القيام به هو توفير الدالتي ادناه
 
 <br>    __len__  –  يتم ارجاع حجم  الداتا من خلال هذه الدالة

 <br>  

  __getitem__  – يتم ارجاع عينه من الداتا وهذه هنا نقوم بارجاع    الفيديو والليبل الخاص به ان كان عنف ا
 و غير عنف
 
 <br>  الكود الهيكيلي للكلاس  هو بالشكل التالي
 
</div>
 
```
from torch.utils.data import Dataset

class FireDataset(Dataset):
    """Fire dataset."""

    def __init__(self, ----):



    def __len__(self):
        return lenght_of_data

    def __getitem__(self, --):

        sample = {'video': torch.from_numpy(video), 'label': torch.from_numpy(label)}


        return sample
```
 
 
<div dir="rtl">
  
 <br>  ان مانود القيام به ف هذه الكلاس هو حين استدعائها ترجع عينة من الفيديو  والليبل   الخاص به سواء كان عنف او غير عنف  هنالك  طريقتين  منطقيتين   الاولى والتي افضلها وسوف استخدمها  وهي اننا نقوم باعطاء الدالة  ملف يحتوي عناوين الملفات الفيديوة مع الليبل الخاص بها عنف او غير عنف   وحين يتم الاستدعاء والعمل  نضع دالة تقوم بقراءة الفيديو   واححد تلو الاخر اثناء التدريب  نحن بهذه الطريقة  سنستخدم رام اقل ولكن سيكون عملية المعالجة تاخذ وقت اكثر  هنا نضحي بالوقت مقارنة بكلفة الرام لكروت الشاشة    - الطريقة الثانية هو قراءة الملفات كلها وخزنها بالذكارة  بحيث حين يتم التدريب الفيديوهات  مخزونة بالذكارة وهنا يكون وقت التدريب اقل ولكن نحتاج ذاكرة رام اكبر لكل من الرام الاساسي وللكارت الشاشة
 <br>  الان لتطبيق الطريقة المقترحة   كل ماعلينا فعلة هو  اولا  يتم توفير مسار ملف الفيديو والليبل المعرف له سواء كان عنف او غير عنف   - ثانيا نحتاج دالة تقوم  بتحويل الفيديو   وقراءة الفيديو من مسارة  وان تعيده بشكل  مصفوفة numpy    - ثالثا نحول الفيديو ونوعه الى تسنور
 
 <br>  وفق اعلاه يمكن كتابة الكلاس كما يلي
 
 <br> 
  
</div>
 
```
from torch.utils.data import Dataset 

class FireDataset(Dataset):
    """Fire dataset."""

    def __init__(self, datas, timesep=30,rgb=3,h=120,w=120):
        """
        Args:
            datas: pandas dataframe contain path to videos files with label of them
            timesep: number of frames
            rgb: number of color chanles
            h: height
            w: width
                 
        """
        self.dataloctions = datas
        self.timesep,self.rgb,self.h,self.w = timesep,rgb,h,w


    def __len__(self):
        return len(self.dataloctions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video = capture(self.dataloctions.iloc[idx, 0],self.timesep,self.rgb,self.h,self.w)
        sample = {'video': torch.from_numpy(video), 'label': torch.from_numpy(np.asarray(self.dataloctions.iloc[idx, 1]))}


        return sample

```

 

<div dir="rtl">
 
<br>  الان  لاحظ اننا بحاجة كتابة دالة تقوم بقراءة اليديو من المسار  المزود لها وارجاعه على شكل مصفوف  بعد  عمل عمليات   تعديل وتحويل  حسب الحاجة 
الدالة التي قمت ببنائها  بالكود التالي
 
</div>
 
```
def capture(filename,timesep,rgb,h,w):
    tmp = []
    frames = np.zeros((timesep,rgb,h,w), dtype=np.float)
    i=0
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    frm = resize(frame,(h, w,rgb))
    frm = np.expand_dims(frm,axis=0)
    frm = np.moveaxis(frm, -1, 1)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    while i < timesep:
        tmp[:] = frm[:]
        rval, frame = vc.read()
        frm = resize(frame,( h, w,rgb))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frm = np.moveaxis(frm, -1, 1)
        frames[i-1][:] = frm # - tmp
        i +=1

    return frames

```
  
## <div dir="rtl">2- بناء  كلاس توزيع  الفيديو على الكونفليوشن الاعتيادية </div>
<div dir="rtl">
 <br>   من المعروف ان اغلب  المودل المدربة   مثل vgg19  , resnet  وغيرها  هيه مبنية على   conv2d  والتي تستقبل مدخل ثلاثي الابعاد  يمثل الطول العرض وقنوات الالوان     وهذه لايمكن ان تستقبل  فيديو مكون من  الفريمات والطول والعرض والقنوات اللونية لذلك لكي نستطيع  تفادي هذا الموضوع نقوم بتزويد الفريمات  واحدة تلو الاخرى للكونفليوشن  نيتورك  هنا  يكون لدينا طريقتين اما عمليه تغيير ابعاد المصفوفة  او عمل      loop  وادخل الفريمات واحد تلو الاخر للكونفليوشن نيتورك
 
 <br>  الطريقة الاولى تستخدم ذاكرة اكثر ولكنها اكفاء بالتعلم   مبنية على  عملة تغيير ابعد المصفوفة فملا لوكان لدينا 10 فيديوهات كل فيديو يتكون من 30 فريم  فيتم التعامل معها كانما 300 عنصر  اي 300 صورة كل فريم يمل صورة 
 
 الكود لهذه الطريقة هو التالي
  
</div>
 
 ```
 # reshape input  to be (batch_size * timesteps, input_size)
 x = x.contiguous().view(batch_size * time_steps, C, H, W)
 # feed to the pre-trained conv model
 x = self.baseModel(x)
 # flatten the output
 x = x.view(x.size(0), -1)
 # make the new correct shape (batch_size , timesteps , output_size)
 x = x.contiguous().view(batch_size , time_steps , x.size(-1))  # this x is now ready to be entred or feed into lstm layer
```
 

<div dir="rtl">
 
<br>  لو اردنا  تطبيقه بكلاس لبناء مودل متكامل يكون كالتالي  مع استخدام المخرجات  ووضعها بطبقة LSTM
  
</div>
 
 
```

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_classes = 1
        dr_rate= 0.2
        pretrained = True
        rnn_hidden_size = 30
        rnn_num_layers = 2
        #get a pretrained vgg19 model ( taking only the cnn layers and fine tun them)
        baseModel = models.vgg19(pretrained=pretrained).features  
        i = 0
        for child in baseModel.children():
            if i < 28:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            i +=1

        num_features = 12800
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True)
        self.fc2 = nn.Linear(30, 256)
        self.fc3 = nn.Linear(256, num_classes)
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        # reshape input  to be (batch_size * timesteps, input_size)
        x = x.contiguous().view(batch_size * time_steps, C, H, W)
        x = self.baseModel(x)
        x = x.view(x.size(0), -1)
        #make output as  ( samples, timesteps, output_size)
        x = x.contiguous().view(batch_size , time_steps , x.size(-1))
        x , (hn, cn) = self.rnn(x)
        x = F.relu(self.fc2(x[:, -1, :])) # get output of the last  lstm not full sequence
        x = self.dropout(x)
        x = self.fc3(x)
        return x 

``` 
 

<div dir="rtl">
 
<br>  الطريقة الثانية باستخدم ال loop    وتمرير  الفريمات واحد تلو الاخر للكونفليوشن نيتورك   وهي اقل استخدام للذاركة ولكن 
تستهلك وت اكبر ووجدت بالتطبيق انها  اقل جودة بالتعلم

 
</div>

 
 ```
        batch_size, time_steps, C, H, W = x.size() #get shape of the input
        output = []
        #loop over each frame
        for i in range(time_steps):
            #input one frame at a time into the basemodel
            x_t = self.baseModel(x[:, i, :, :, :])
            # Flatten the output
            x_t = x_t.view(x_t.size(0), -1)
            #make a list of tensors for the given smaples 
            output.append(x_t)
        #end loop  
        #make output as  ( samples, timesteps, output_size)
        x = torch.stack(output, dim=0).transpose_(0, 1)   # this x is now ready to be entred or feed into lstm layer
```
 

<div dir="rtl">
 
<br>  لو اردنا  تطبيقه بكلاس لبناء مودل متكامل يكون كالتالي  مع استخدام المخرجات  ووضعها بطبقة LSTM
 
</div>


 

```

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        num_classes = 1
        dr_rate= 0.2
        pretrained = True
        rnn_hidden_size = 30
        rnn_num_layers = 2
        baseModel = models.vgg19(pretrained=pretrained).features
        i = 0
        for child in baseModel.children():
            if i < 28:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            i +=1

        num_features = 12800
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True)
        self.fc1 = nn.Linear(30, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        output = []
        for i in range(time_steps):
            #input one frame at a time into the basemodel
            x_t = self.baseModel(x[:, i, :, :, :])
            # Flatten the output
            x_t = x_t.view(x_t.size(0), -1)
            output.append(x_t)
        #end loop
        #make output as  ( samples, timesteps, output_size)
        x = torch.stack(output, dim=0).transpose_(0, 1)
        output = None # clear var to reduce data  in memory
        x_t = None  # clear var to reduce data  in memory
        x , (hn, cn) = self.rnn(x)
        x = F.relu(self.fc1(x[:, -1, :])) # get output of the last  lstm not full sequence
        x = self.dropout(x)
        x = self.fc2(x)
        return x 
```


 
<div dir="rtl">
 
<br>  لتسهيل العمل وكتابة كود نضيف ساقوم بعمل كلاس   يقوم بالعمليتين حسب اختيارنا اثناء البناء  وايضا يسهل لناعملية استدعاء الكلاس باي مكان نريد
 
</div>


 
```
class TimeWarp(nn.Module):
    def __init__(self, baseModel, method='sqeeze'):
        super(TimeWarp, self).__init__()
        self.baseModel = baseModel
        self.method = method
 
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        if self.method == 'loop':
            output = []
            for i in range(time_steps):
                #input one frame at a time into the basemodel
                x_t = self.baseModel(x[:, i, :, :, :])
                # Flatten the output
                x_t = x_t.view(x_t.size(0), -1)
                output.append(x_t)
            #end loop
            #make output as  ( samples, timesteps, output_size)
            x = torch.stack(output, dim=0).transpose_(0, 1)
            output = None # clear var to reduce data  in memory
            x_t = None  # clear var to reduce data  in memory
        else:
            # reshape input  to be (batch_size * timesteps, input_size)
            x = x.contiguous().view(batch_size * time_steps, C, H, W)
            x = self.baseModel(x)
            x = x.view(x.size(0), -1)
            #make output as  ( samples, timesteps, output_size)
            x = x.contiguous().view(batch_size , time_steps , x.size(-1))
        return x

```


<div dir="rtl">
 
لو اردنا استخدام الكلاس اعلاه    يمكن ذلك من خلال الكود التوضيحي التالي 
</div> 



```
baseModel = models.vgg19(pretrained=pretrained).features

model = nn.Sequential(TimeWarp(baseModel))
```

 
<div dir="rtl">
<br>
الان نحن بالتاكيد نريد استخدام    LSTM    وجدت ان  هنالك مشكلة لدى البعض عملية استخدام ال LSTM  في داخل nn.Sequential
حيث وجدت احدهم يسال عن ذلك بالستاك اوفر فلو وقمت باجابته هناك من الرابط التالي
https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module

عموما الفكرة هو ببالتحكم بمخرجات الكلاس الخاصة بالLSTM
وقمت بها بالطريقة التالية 
  
</div>

```
class extractlastcell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]
```


<div dir="rtl">
<br> 
الكلاس تستقبل مخرجات الLSTM  كمدخل لها  ونقوم   حسب حاجتنا بارجاع المخرج المناسب  نحن هن نقوم بارجاع  اخر مخرج من اخر  cell


الان الكود الكامل لاستخدام الكلاسين وبناء  مودل كامل هو كالتالي
مع استخدم transffer learning 
 
</div>
 

```
# Create model

num_classes = 1
dr_rate= 0.2
pretrained = True
rnn_hidden_size = 30
rnn_num_layers = 2
baseModel = models.vgg19(pretrained=pretrained).features
i = 0
for child in baseModel.children():
    if i < 28:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True
    i +=1

num_features = 12800

# Example of using Sequential
model = nn.Sequential(TimeWarp(baseModel),
                       nn.Dropout(dr_rate),
                      nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True),
                      extractlastcell(),
                        nn.Linear(30, 256),
                      nn.ReLU(),
                       nn.Dropout(dr_rate),
 nn.Linear(256, num_classes)

        )
```

<div dir="rtl">
 
هنا نكون قد انتهينا     من العناصر المهمه وقد قمت ببناء وتدريب مودل ورفعه على شكل  Rest API  موجود من الرابط التالي مع الاوزان الخاصة بالتدريب
 
https://github.com/mamonraab/violance-detection-in-video-with-pytroch/tree/main/flaskapp
ويمكنكم ان تقومو بتدريب مودل خاص بكم باستخدام ماتم شرحة بهذا المقال 

مع التحية والسلام
اود ان اشير الى انني    قد قمت قبل سنة بكتابة ورقة علمية متخصصة بهذا الموضوع   ولن الكود البرمجي لها كان باستخدم التنسورفلو وبمعمارية مختلفة عن  المذكورة    بهذا المقال
اذا احببت ان تطلع على الورقة العملية يمكنك من الرابط التالي
https://www.researchgate.net/publication/336156932_Robust_Real-Time_Violence_Detection_in_Video_Using_CNN_And_LSTM
</div>


