#!/usr/bin/env python
# coding: utf-8

# # Bài tập 2

# # Mục lục
# 1. [Giới thiệu](#gioithieu)
# 
# 2. [Phân tích dữ liệu](#dataset)
# 
#     2.1 [Thêm thư viện](#library)
#     
#     2.2 [Thêm đường dẫn dữ liệu](#data)
#     
#     2.3 [Mô tả dữ liệu](#description)
#     
#     2.4 [Mô tả dữ liệu số](#numerical)
# 
# 3. [Các yếu tố có thể ảnh hưởng đến thu nhập](#question1)
# 
#     3.1 [Age](#age1)
#     
#     3.2 [Capital-gain](#capitalgain)
#     
#     3.3 [Capital-loss](#capitalloss)
#     
#     3.4 [Educational-num](#educationalnum)
#     
#     3.5 [Hour-per-week](#hourperweek)
# 
# 4. [Dự đoán về đặc điểm của một người có thu nhập cao](#question2)

# ## 1. Giới thiệu <a name="introduction"></a>
# - Đây là dữ liệu thu nhập hằng năm kèm theo các yếu tố như trình độ học vấn, độ tuổi, giới tính, nghề nghiệp,...Tập dữ liệu này được lấy từ tập databases machine learning của kho lưu trữ UCI http://www.cs.toronto.edu/~delve/data/adult/desc.html. 
# 
# - Phần trình bày bao gồm 3 phần. Đầu tiên là mô tả dữ liệu, sau đó là tìm các yếu tố có thể ảnh hưởng đến thu nhập, cuối cùng là dự đoán một người có thu nhập cao thì có thể sẽ có những đặc điểm nào.
# 
# - Để thuận tiện cho việc mô tả dữ liệu, phần trình bày sẽ sử dụng ngôn ngữ lập trình python chạy trên jupyter notebook.

# ## 2. Phân tích dữ liệu<a name="dataset"></a>

# #### Thêm thư viện<a name="library"></a>

# In[1]:


import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind


# #### Thêm đường dẫn đến tập dữ liệu<a name="data"></a>

# In[30]:


url = "D:\Hoang\SauDaiHoc\PhuongPhapNCKH\Bai 2\income1.csv"
data = pd.read_csv(url)


# #### Mô tả dữ liệu<a name="description"></a>

# In[31]:


# Kích thước của tập dữ liệu
print("Kích thước dữ liệu là: {}".format(data.shape))


# In[32]:


# Tên các cột của tập dữ liệu
data.columns


# In[33]:


# 5 dòng đầu của tập dữ liệu
data.head()


# In[34]:


# Kiểu dữ liệu của các cột
data.info()


# In[82]:


# Mô tả tập dữ liệu
perc = [.25, .5, .75] # percentile list
data.describe(percentiles = perc, include = None, exclude = None)


# - count là số lượng
# - mean là giá trị trung bình
# - std là độ lệch chuẩn
# - min, max là giá trị nhỏ nhất và lớn nhất
# - 25%,50%,75% là giá trị chiếm số lượng 25%,50%,75% trong tổng thể

# #### Trực quan hóa dữ liệu số<a name="numerical"></a>

# In[36]:


data.hist(figsize=(15,15))
plt.show()


# #### Nhận xét
# - Độ tuổi khảo sát phân bố đều từ khoảng 18 đến 45 tuổi, sau 45 tuổi bắt đầu giảm dần
# - Capital-gain và Capital-loss tập trung chủ yếu là 0
# - Ở Education-num, phân bố tập trung chủ yếu trong khoảng [9,11] và 14
# - Ở fnlwgt, phân bố tập trung chủ yếu từ 0 dến 300000, ở mức 400000 chiếm một số lượng nhỏ và từ 420000 chiếm số lượng rất ít
# - Ở hours-per-week, tập trung chủ yếu là 40

# ## 3. Các yếu tố có thể ảnh hưởng đến thu nhập<a name="question1"></a>

# Phần này sẽ chọn 5 yếu tố gồm age, capital-gain, capital-loss, educational-num, hours-per-week là các biến số và sử dụng independent t-test để kiểm tra xem những yếu tố nào ảnh hưởng đến thu nhập.

# ### 1.1 Age<a name="age1"></a>

# In[9]:


data['income'].value_counts()


# In[11]:


print("Phần trăm người có thu nhập lớn hơn 50k là {:.1f}% và phần trăm người có thu nhập nhỏ hơn {:.1f}%".format(
    data[data['income'] == '>50K'].shape[0] / data.shape[0]*100,
    data[data['income'] == '<=50K'].shape[0] / data.shape[0]*100))


# In[14]:


plt.hist(data.age[data.income == '>50K'], label=['Age, >50K'])
plt.legend(loc='upper right')
plt.xlabel("Age")
plt.ylabel("Count")
np.var(data.age[data.income == '>50K'])


# In[15]:


plt.hist(data.age[data.income == '<=50K'], label=['Age,<=50K'])
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.age[data.income == '<=50K'])


# #### Kiểm tra phương sai bằng nhau

# Giả thuyết null: Các phương sai là bằng nhau

# In[16]:


scipy.stats.levene(data.age[data.income == '>50K'],data.age[data.income == '<=50K'])


# pvalue<0.05. Các phương sai là không bằng nhau

# #### t-test
# Giả thuyết null H0: Không có sự khác biệt đáng kể giữa tuổi trong 2 nhóm thu nhập

# In[60]:


ttest_ind(data.age[data.income == '>50K'],data.age[data.income == '<=50K'],equal_var=False,nan_policy='omit')


# pvalue<0.05. Phủ định H0. Có sự khác biệt giữa tuổi trong 2 nhóm thu nhập

# ### 1.2 Capital-gain <a name="capitalgain"></a>

# In[37]:


plt.hist(data.capitalgain[data.income == '>50K'], label =['capital-gain, >50K'])
plt.xlabel("capital-gain")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.capitalgain[data.income == '>50K'])


# In[41]:


plt.hist(data.capitalgain[data.income == '<=50K'], label=['capital-gain, <=50K'])
plt.xlabel("capital-loss")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.capitalgain[data.income == '<=50K'])


# #### Kiểm tra phương sai bằng nhau

# Giả thuyết null: Các phương sai là bằng nhau

# In[42]:


scipy.stats.levene(data.capitalgain[data.income == '>50K'],data.capitalgain[data.income == '<=50K'])


# pvalue<0.05. Các phương sai là không bằng nhau

# #### t-test
# Giả thuyết null H0: Không có sự khác biệt đáng kể trong capital-gain giữa hai nhóm thu nhập cao và thấp 

# In[43]:


ttest_ind(data.capitalgain[data.income == '>50K'],data.capitalgain[data.income == '<=50K'],equal_var=False,nan_policy='omit')


# pvalue<0.05. Phủ định H0. Có sự khác biệt trong capital-gain giữa hai nhóm thu nhập cao và thấp

# ### 1.3 Capital-loss<a name="capitalloss"></a>

# In[45]:


plt.hist(data.capitalloss[data.income == '>50K'], label=['capital-loss, >50K'])
plt.xlabel("capital-loss")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.capitalloss[data.income == '>50K'])


# In[46]:


plt.hist(data.capitalloss[data.income == '<=50K'], label=['capital-loss, <=50K'])
plt.xlabel("capital-loss")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.capitalloss[data.income == '<=50K'])


# #### Kiểm tra phương sai bằng nhau

# Giả thuyết null: Các phương sai là bằng nhau

# In[48]:


scipy.stats.levene(data.capitalloss[data.income == '>50K'],data.capitalloss[data.income == '<=50K'])


# pvalue<0.05. Các phương sai là không bằng nhau

# #### t-test
# Giả thuyết H0: Không có sự khác biệt đáng kể trong capital-loss giữa hai nhóm thu nhập cao và thấp

# In[61]:


ttest_ind(data.capitalloss[data.income == '>50K'],data.capitalloss[data.income == '<=50K'],equal_var=False,nan_policy='omit')


# pvalue<0.05. Phủ định H0. Có sự khác biệt trong capital-loss giữa hai nhóm thu nhập cao và thấp

# ### 1.4 Educational-num <a name="educationalnum"></a>

# In[52]:


plt.hist(data.educationalnum[data.income == '>50K'], label=['educational-num, >50K'])
plt.xlabel("capital-loss")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.educationalnum[data.income == '>50K'])


# In[56]:


plt.hist(data.educationalnum[data.income == '<=50K'], label=['educational-num, <=50K'])
plt.xlabel("educational-num")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.educationalnum[data.income == '<=50K'])


# ### Kiểm tra phương sai bằng nhau

# In[64]:


scipy.stats.levene(data.educationalnum[data.income == '>50K'],data.educationalnum[data.income == '<=50K'])


# pvalue<0.05. Các phương sai là không bằng nhau

# ### t-test

# In[62]:


ttest_ind(data.educationalnum[data.income == '>50K'],data.educationalnum[data.income == '<=50K'],equal_var=False,nan_policy='omit')


# pvalue<0.05. Có sự khác biệt trong educaltion-num giữa hai nhóm thu nhập cao và thấp

# ### 1.5 Hours-per-week <a name="hourperweek"></a>

# In[67]:


plt.hist(data.hoursperweek[data.income == '>50K'], label=['Hour-per-week, >50K'])
plt.xlabel("Hours-per-week")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.hoursperweek[data.income == '>50K'])


# In[71]:


plt.hist(data.hoursperweek[data.income == '<=50K'], label=['Hour-per-week, <=50K'])
plt.xlabel("Hours-per-week")
plt.ylabel("Count")
plt.legend(loc='upper right')
np.var(data.hoursperweek[data.income == '<=50K'])


# ### Kiểm tra phương sai bằng nhau

# In[73]:


scipy.stats.levene(data.hoursperweek[data.income == '>50K'],data.hoursperweek[data.income == '<=50K'])


# pvalue<0.05. Phương sai không bằng nhau

# ### t-test

# In[75]:


ttest_ind(data.hoursperweek[data.income == '>50K'],data.hoursperweek[data.income == '<=50K'],equal_var=False,nan_policy='omit')


# pvalue<0.05. Có sự khác biệt trong hours-per-week giữa hai nhóm thu nhập cao và thấp

# ### Trả lời:
# - Cả 5 yếu tố Age, Capital-gain, Capital-loss, Educational-num và Hours-per-week ảnh hưởng đến tỉ lệ thu nhập cao và thấp

# ## 4. Dự đoán về đặc điểm của một người có thu nhập cao<a name="question2"></a>

# Phần này sẽ tìm giá trị xuất hiện nhiều nhất của Age, Capital-gain, Capital-loss, Educational-num và Hours-per-week trong 3 nhóm gồm tổng thể, thu nhập trên 50k và thu nhập dưới 50k. Sau đó sẽ so sánh các giá trị này để dự đoán đặc điểm của một người có thu nhập cao

# In[77]:


#Phân chia dữ liệu thành 2 nhóm thu nhập cao và thu nhập thấp
thunhapcao = data.loc[data['income'] == ">50K"]
thunhapthap = data.loc[data['income'] == "<=50K"]


# In[81]:


#Số lần xuất hiện nhiều nhất trong tổng thể
print("Trong tổng thể:")
print(" - Số lần xuất hiện nhiều nhất của Age là: {}".format(np.median(data.age)))
print(" - Số lần xuất hiện nhiều nhất của Capital-gain là: {}".format(np.median(data.capitalgain)))
print(" - Số lần xuất hiện nhiều nhất của Capital-loss là: {}".format(np.median(data.capitalloss)))
print(" - Số lần xuất hiện nhiều nhất của Educational-num là: {}".format(np.median(data.educationalnum)))
print(" - Số lần xuất hiện nhiều nhất của Hours-per-week: {}".format(np.median(data.hoursperweek)))
#Số lần xuất hiện nhiều nhất trong nhóm thu nhập cao
print("Trong nhóm thu nhập cao:")
print(" - Số lần xuất hiện nhiều nhất của Age là: {}".format(np.median(thunhapcao.age)))
print(" - Số lần xuất hiện nhiều nhất của Capital-gain là: {}".format(np.median(thunhapcao.capitalgain)))
print(" - Số lần xuất hiện nhiều nhất của Capital-loss là: {}".format(np.median(thunhapcao.capitalloss)))
print(" - Số lần xuất hiện nhiều nhất của Educational-num là: {}".format(np.median(thunhapcao.educationalnum)))
print(" - Số lần xuất hiện nhiều nhất của Hours-per-week: {}".format(np.median(thunhapcao.hoursperweek)))
#Số lần xuất hiện nhiều nhất trong nhóm thu nhập thấp
print("Trong nhóm thu nhập thấp:")
print(" - Số lần xuất hiện nhiều nhất của Age là: {}".format(np.median(thunhapthap.age)))
print(" - Số lần xuất hiện nhiều nhất của Capital-gain là: {}".format(np.median(thunhapthap.capitalgain)))
print(" - Số lần xuất hiện nhiều nhất của Capital-loss là: {}".format(np.median(thunhapthap.capitalloss)))
print(" - Số lần xuất hiện nhiều nhất của Educational-num là: {}".format(np.median(thunhapthap.educationalnum)))
print(" - Số lần xuất hiện nhiều nhất của Hours-per-week: {}".format(np.median(thunhapthap.hoursperweek)))


# - Có sự khác biệt trong số lần xuất hiện nhiều nhất của Age và Educational-num trong hai nhóm thu nhập cao và thu nhập thấp.
# - Nhưng không có sự khác biệt trong Captital-gain, Capital-loss, Hours-per-week.

# ### Trả lời:
# - Một người có thu nhập cao thì độ tuổi là 43 và Educational-num là 12
