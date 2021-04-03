
# coding: utf-8

# In[288]:


#!/usr/bin/env python
# coding: utf-8

# In[108]:


import os
os.chdir("C:\python")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DF = pd.read_csv('awesome_calc.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)
DF2 = pd.read_csv('awesome_calc.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)
DF3 = pd.read_csv('awesome_calc.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)#для реальных значений
DF5 = pd.read_csv('awesome_calc.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)#для среднего



DF1 = pd.read_csv('awesome_calc2.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)
DF4 = pd.read_csv('awesome_calc2.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)
DF6 = pd.read_csv('awesome_calc2.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)#для реальных значений
DF7 = pd.read_csv('awesome_calc2.csv', encoding ='windows-1251', sep=";", header = 0, index_col=0)#для среднего

#DF=DF.drop(['Arts & Humanities','Engineering & Technology','Life Sciences & Medicine','Natural Sciences','Social Sciences & Management'])
#DF.dropna(inplace = True)
#df.drop(['Cochice', 'Pima'])
#pd.options.display.float_format = "{:.00f}".format



# In[118]:




# In[ ]:


# In[289]:


DF['1.1'] = (((DF['1.1'].str.replace(',','.').astype(float)/69)**0.5)*100).clip(upper=100)
DF['1.2'] = (((DF['1.2'].str.replace(',','.').astype(float)/26)**0.5)*100).clip(upper=100)
DF['1.4'] = (((DF['1.4'].str.replace(',','.').astype(float)/75)**0.5)*100).clip(upper=100)
DF['1.5'] = (((DF['1.5'].astype(float)/40)**0.5)*100).clip(upper=100)

DF['1.7'] = (((DF['1.7'].astype(float)/100)**0.5)*100).clip(upper=100)
DF['1.9'] = (((DF['1.9'].str.replace(',','.').astype(float)/20)**0.5)*100).clip(upper=100)
DF['1.10'] = (((DF['1.10'].str.replace(',','.').astype(float)/100)**0.5)*100).clip(upper=100)
DF['3.5'] = (((DF['3.5'].str.replace(',','.').astype(float)/2)**-0.5)*100).clip(upper=100)
DF['4.1'] = (((DF['4.1'].astype(float)/15.9)**0.5)*100).clip(upper=100)
DF['6.2'] = (((DF['6.2'].str.replace(',','.').astype(float)/80)**0.5)*100).clip(upper=100)
DF['6.3'] = (((DF['6.3'].astype(float)/30)**0.5)*100).clip(upper=100)
DF['1.8'] = (((DF['1.8'].str.replace(',','.').astype(float)/30)**0.5)*100).clip(upper=100)


DF = DF.rename(index={'Инженерно-технический институт': 'Инж-тех',
                 'Институт гуманитарных наук': 'ИГН',
                 'Институт живых систем': 'ИЖС',
                 'Институт образования': 'ИнОбр.',
                 'Институт природопользования территориального развития и градостроительства': 'ИПТРиГ',
                 'Институт рекреации, туризма и физической культуры': 'ИРТиФК',
                 'Институт физико-математических наук и информационных технологий': 'Физ-мат',
                 'Институт экономики и менеджмента': 'ИЭиМ',
                 'Медицинский институт': 'МедИн',
                 'Юридический институт': 'ЮридИн'})

DF3 = DF3.rename(index={'Инженерно-технический институт': 'Инж-тех',
                 'Институт гуманитарных наук': 'ИГН',
                 'Институт живых систем': 'ИЖС',
                 'Институт образования': 'ИнОбр.',
                 'Институт природопользования территориального развития и градостроительства': 'ИПТРиГ',
                 'Институт рекреации, туризма и физической культуры': 'ИРТиФК',
                 'Институт физико-математических наук и информационных технологий': 'Физ-мат',
                 'Институт экономики и менеджмента': 'ИЭиМ',
                 'Медицинский институт': 'МедИн',
                 'Юридический институт': 'ЮридИн'})


# In[290]:


display(DF.round(2))
display(DF1.round(2))


# In[291]:


DF1['1.7'] = (((DF1['1.7'].astype(float)/100)**0.5)*100).clip(upper=100)
DF1['2.1'] = (((DF1['2.1'].str.replace(',','.').astype(float)/1)**0.5)*100).clip(upper=100)
DF1['2.2'] = (((DF1['2.2'].str.replace(',','.').astype(float)/0.8)**0.5)*100).clip(upper=100)
DF1['2.3'] = (((DF1['2.3'].str.replace(',','.').astype(float)/0.25)**0.5)*100).clip(upper=100)

DF1['2.4'] = (((DF1['2.4'].str.replace(',','.').astype(float)/4)**0.5)*100).clip(upper=100)
DF1['2.5'] = (((DF1['2.5'].str.replace(',','.').astype(float)/0.053)**0.5)*100).clip(upper=100)
DF1['2.7'] = (((DF1['2.7'].str.replace(',','.').astype(float)/5.3)**0.5)*100).clip(upper=100)
DF1['3.1'] = (((DF1['3.1'].str.replace(',','.').astype(float)/900)**0.5)*100).clip(upper=100)
DF1['3.3'] = (((DF1['3.3'].str.replace(',','.').astype(float)/178)**0.5)*100).clip(upper=100)
DF1['6.2'] = (((DF1['6.2'].str.replace(',','.').astype(float)/80)**0.5)*100).clip(upper=100)
DF1['6.3'] = (((DF1['6.3'].astype(float)/30)**0.5)*100).clip(upper=100)
DF1['6.5'] = (((DF1['6.5'].astype(float)/40)**0.5)*100).clip(upper=100)


DF1 = DF1.rename(index={'Инженерно-технический институт': 'Инж-тех',
                 'Институт гуманитарных наук': 'ИГН',
                 'Институт живых систем': 'ИЖС',
                 'Институт образования': 'ИнОбр.',
                 'Институт природопользования территориального развития и градостроительства': 'ИПТРиГ',
                 'Институт рекреации, туризма и физической культуры': 'ИРТиФК',
                 'Институт физико-математических наук и информационных технологий': 'Физ-мат',
                 'Институт экономики и менеджмента': 'ИЭиМ',
                 'Медицинский институт': 'МедИн',
                 'Юридический институт': 'ЮридИн'})

DF4 = DF4.rename(index={'Инженерно-технический институт': 'Инж-тех',
                 'Институт гуманитарных наук': 'ИГН',
                 'Институт живых систем': 'ИЖС',
                 'Институт образования': 'ИнОбр.',
                 'Институт природопользования территориального развития и градостроительства': 'ИПТРиГ',
                 'Институт рекреации, туризма и физической культуры': 'ИРТиФК',
                 'Институт физико-математических наук и информационных технологий': 'Физ-мат',
                 'Институт экономики и менеджмента': 'ИЭиМ',
                 'Медицинский институт': 'МедИн',
                 'Юридический институт': 'ЮридИн'})


# In[292]:


DF.round(3)


# In[293]:


DF1.round(3)


# In[294]:


#замена строкового формата данных на float
#!!!!перевести все значения в оригинальной таблице в float!!!!!!, поправить код

DF3['1.1'] = (DF3['1.1'].str.replace(',','.').astype(float))
DF3['1.2'] = (DF3['1.2'].str.replace(',','.').astype(float))
DF3['1.4'] = (DF3['1.4'].str.replace(',','.').astype(float))
DF3['1.5'] = (DF3['1.5'].astype(float))
DF3['1.7'] = (DF3['1.7'].astype(float))
DF3['1.9'] = (DF3['1.9'].str.replace(',','.').astype(float))
DF3['1.10'] = (DF3['1.10'].str.replace(',','.').astype(float))
DF3['3.5'] = (DF3['3.5'].str.replace(',','.').astype(float))
DF3['4.1'] = (DF3['4.1'].astype(float))
DF3['6.2'] = (DF3['6.2'].str.replace(',','.').astype(float))
DF3['6.3'] = (DF3['6.3'].astype(float))
DF3['1.8'] = (DF3['1.8'].str.replace(',','.').astype(float))


# In[295]:


DF6


# In[296]:


#замена строкового формата данных на float
#!!!!перевести все значения в оригинальной таблице в float!!!!!!, поправить код

DF6['1.7'] = (DF6['1.7'].astype(float))
DF6['2.1'] = (DF6['2.1'].str.replace(',','.').astype(float))
DF6['2.2'] = (DF6['2.2'].str.replace(',','.').astype(float))
DF6['2.3'] = (DF6['2.3'].str.replace(',','.').astype(float))
DF6['2.4'] = (DF6['2.4'].str.replace(',','.').astype(float))
DF6['2.5'] = (DF6['2.5'].str.replace(',','.').astype(float))
DF6['2.7'] = (DF6['2.7'].str.replace(',','.').astype(float))
DF6['3.1'] = (DF6['3.1'].str.replace(',','.').astype(float))
DF6['3.3'] = (DF6['3.3'].str.replace(',','.').astype(float))
DF6['6.2'] = (DF6['6.2'].str.replace(',','.').astype(float))
DF6['6.3'] = (DF6['6.3'].astype(float))
DF6['6.5'] = (DF6['6.5'].astype(float))


# In[297]:


#расчет среднего по каждому показателю
#!!!!перевести все значения в оригинальной таблице в float!!!!!!, поправить код

DF5['1.1'] = (DF5['1.1'].str.replace(',','.').astype(float)).mean()
DF5['1.2'] = (DF5['1.2'].str.replace(',','.').astype(float)).mean()
DF5['1.4'] = (DF5['1.4'].str.replace(',','.').astype(float)).mean()
DF5['1.5'] = (DF5['1.5'].astype(float)).mean()
DF5['1.7'] = (DF5['1.7'].astype(float)).mean()
DF5['1.9'] = (DF5['1.9'].str.replace(',','.').astype(float)).mean()
DF5['1.10'] = (DF5['1.10'].str.replace(',','.').astype(float)).mean()
DF5['3.5'] = (DF5['3.5'].str.replace(',','.').astype(float)).mean()
DF5['4.1'] = (DF5['4.1'].astype(float)).mean()
DF5['6.2'] = (DF5['6.2'].str.replace(',','.').astype(float)).mean()
DF5['6.3'] = (DF5['6.3'].astype(float)).mean()
DF5['1.8'] = (DF5['1.8'].str.replace(',','.').astype(float)).mean()
xxss = DF5['1.1']
xxss2 = DF5['1.2']
xxss3 = DF5['1.4']
xxss4 = DF5['1.5']
xxss5 = DF5['1.7']
xxss6 = DF5['1.9']
xxss7 = DF5['1.10']
xxss8 = DF5['3.5']
xxss9 = DF5['4.1']
xxss10 = DF5['6.2']
xxss11 = DF5['6.3']
xxss12 = DF5['1.8']


# In[298]:


xs = DF.index
xs2 = DF.index
xs3 = DF.index
xs4 = DF.index
xs5 = DF.index
xs6 = DF.index
xs7 = DF.index
xs8 = DF.index
xs9 = DF.index
xs10 = DF.index
xs11 = DF.index
xs12 = DF.index

xss = DF.index
xss2 = DF.index
xss3 = DF.index
xss4 = DF.index
xss5 = DF.index
xss6 = DF.index
xss7 = DF.index
xss8 = DF.index
xss9 = DF.index
xss10 = DF.index
xss11 = DF.index
xss12 = DF.index

ys = DF['1.1']
ys2 = DF['1.2']
ys3 = DF['1.4']
ys4 = DF['1.5']
ys5 = DF['1.7']
ys6 = DF['1.9']
ys7 = DF['1.10']
ys8 = DF['3.5']
ys9 = DF['4.1']
ys10 = DF['6.2']
ys11 = DF['6.3']
ys12 = DF['1.8']

yss = DF3['1.1']
yss2 = DF3['1.2']
yss3 = DF3['1.4']
yss4 = DF3['1.5']
yss5 = DF3['1.7']
yss6 = DF3['1.9']
yss7 = DF3['1.10']
yss8 = DF3['3.5']
yss9 = DF3['4.1']
yss10 = DF3['6.2']
yss11 = DF3['6.3']
yss12 = DF3['1.8']

xss, yss = zip(*sorted(zip(yss,xss)))
xss2, yss2 = zip(*sorted(zip(yss2,xss2)))
xss3, yss3 = zip(*sorted(zip(yss3,xss3)))
xss4, yss4 = zip(*sorted(zip(yss4,xss4)))
xss5, yss5 = zip(*sorted(zip(yss5,xss5)))
xss6, yss6 = zip(*sorted(zip(yss6,xss6)))
xss7, yss7 = zip(*sorted(zip(yss7,xss7)))
xss8, yss8 = zip(*sorted(zip(yss8,xss8)))
xss9, yss9 = zip(*sorted(zip(yss9,xss9)))
xss10, yss10 = zip(*sorted(zip(yss10,xss10)))
xss11, yss11 = zip(*sorted(zip(yss11,xss11)))
xss12, yss12 = zip(*sorted(zip(yss12,xss12)))


xs, ys = zip(*sorted(zip(ys, xs)))
xs2, ys2 = zip(*sorted(zip(ys2, xs2)))
xs3, ys3 = zip(*sorted(zip(ys3, xs3)))
xs4, ys4 = zip(*sorted(zip(ys4, xs4)))
xs5, ys5 = zip(*sorted(zip(ys5, xs5)))
xs6, ys6 = zip(*sorted(zip(ys6, xs6)))
xs7, ys7 = zip(*sorted(zip(ys7, xs7)))
xs8, ys8 = zip(*sorted(zip(ys8, xs8)))
xs9, ys9 = zip(*sorted(zip(ys9, xs9)))
xs10, ys10 = zip(*sorted(zip(ys10, xs10)))
xs11, ys11 = zip(*sorted(zip(ys11, xs11)))
xs12, ys12 = zip(*sorted(zip(ys12, xs12)))



fig, ((ax1, ax2, ax3, ax4), (ax5,ax6,ax7,ax8), (ax9,ax10, ax11,ax12) ) = plt.subplots(3, 4, figsize=(20,17)) 
fig.suptitle('Рейтинг институтов БФУ Декабрь 2020 (баллы, факт %, среднее %)',  fontsize=20) 

ax1.bar(ys,xs)
ax1.plot(ys,xs, marker='o')
ax1.plot(ys,xxss, c='gold')
ax1.bar(ys,xss)
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('Доля контракт', fontsize=12)
ax1.text(1, 5, r'ЦелПоказ = 69%')

ax2.bar(ys2,xs2)
ax2.plot(ys2,xs2, marker='o')
ax2.plot(ys,xxss2, c='gold')
ax2.bar(ys2,xss2)
ax2.tick_params(axis='x', labelrotation=90)
ax2.set_ylabel('Доля магистров и аспирантов', fontsize=12)
ax2.text(1, 5, r'ЦелПоказ = 26%')

ax3.bar(ys3,xs3)
ax3.plot(ys3,xs3, marker='o')
ax3.plot(ys,xxss3, c='gold')
ax3.bar(ys3,xss3)
ax3.tick_params(axis='x', labelrotation=90)
ax3.set_ylabel('ЕГЭ, бюджет', fontsize=12)
ax3.text(1, 5, r'ЦелПоказ = 75%')

ax4.bar(ys4,xs4)
ax4.plot(ys4,xs4, marker='o')
ax4.plot(ys,xxss4, c='gold')
ax4.bar(ys4,xss4)
ax4.tick_params(axis='x', labelrotation=90)
ax4.set_ylabel('Доля магистров и аспирантов из других вузов', fontsize=12)
ax4.text(1, 5, r'ЦелПоказ = 40%')

ax5.bar(ys5,xs5)
ax5.plot(ys5,xs5, marker='o')
ax5.plot(ys,xxss5, c='gold')
ax5.bar(ys5,xss5)
ax5.tick_params(axis='x', labelrotation=90)
ax5.set_ylabel('Доля преподавателей, использующих ЭОР', fontsize=12)
ax5.text(1, 5, r'ЦелПоказ = 100%')

ax6.bar(ys6,xs6)
ax6.plot(ys6,xs6, marker='o')
ax6.plot(ys,xxss6, c='gold')
ax6.bar(ys6,xss6)
ax6.tick_params(axis='x', labelrotation=90)
ax6.set_ylabel('Доля программ с онлайн', fontsize=12)
ax6.text(1, 5, r'ЦелПоказ = 20%')

ax7.bar(ys7,xs7)
ax7.plot(ys7,xs7, marker='o')
ax7.plot(ys,xxss7, c='gold')
ax7.bar(ys7,xss7)
ax7.tick_params(axis='x', labelrotation=90)
ax7.set_ylabel('Доля общей занятости выпускников', fontsize=12)
ax7.text(1, 5, r'ЦелПоказ = 100%')

ax8.bar(ys8,xs8)
ax8.plot(ys8,xs8, marker='o')
ax8.plot(ys,xxss8, c='gold')
ax8.bar(yss8,xss8) #yss8 = порядок задолженности
ax8.tick_params(axis='x', labelrotation=90)
ax8.set_ylabel('Задолженность', fontsize=12)
ax8.text(1, 1, r'ЦелПоказ = 2%(макс)')

ax9.bar(ys9,xs9)
ax9.plot(ys9,xs9, marker='o')
ax9.plot(ys,xxss9, c='gold')
ax9.bar(ys9,xss9)
ax9.tick_params(axis='x', labelrotation=90)
ax9.set_ylabel('Доля иностранных студентов', fontsize=12)
ax9.text(1, 1, r'ЦелПоказ = 15,9%')

ax10.bar(ys10,xs10)
ax10.plot(ys10,xs10, marker='o')
ax10.plot(ys,xxss10, c='gold')
ax10.bar(ys10,xss10)
ax10.tick_params(axis='x', labelrotation=90)
ax10.set_ylabel('Доля остепененных НПР', fontsize=12)
ax10.text(1, 5, r'ЦелПоказ = 80%')


ax11.bar(ys11,xs11)
ax11.plot(ys11,xs11, marker='o')
ax11.plot(ys,xxss11, c='gold')
ax11.bar(ys11,xss11)
ax11.tick_params(axis='x', labelrotation=90)
ax11.set_ylabel('Доля НПР до 35 лет', fontsize=12)
ax11.text(1, 5, r'ЦелПоказ = 30%')


ax12.bar(ys12,xs12)
ax12.plot(ys12,xs12, marker='o')
ax12.plot(ys,xxss12, c='gold')
ax12.bar(ys12,xss12)
ax12.tick_params(axis='x', labelrotation=90)
ax12.set_ylabel('Доля совместных программ', fontsize=12)
ax12.text(1, 5, r'ЦелПоказ = 30%')


for x,y in zip(xs,ys):
    
    label = "{:.1f}".format(x)
    
    ax1.annotate(label, # this is the text
                 (y,x), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

for x,y in zip(xss,ys):

    label = "{:.1f}".format(x)

    ax1.annotate(label, # this is the text
                 (y,x), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
for x,y in zip(xs2,ys2):

    label = "{:.0f}".format(x)

    ax2.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 

for x,y in zip(xss2,ys2):

    label = "{:.1f}".format(x)
        
    ax2.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 

for x,y in zip(xs3,ys3):

    label = "{:.0f}".format(x)

    ax3.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 

for x,y in zip(xss3,ys3):

    label = "{:.1f}".format(x)

    ax3.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 
    
for x,y in zip(xs4,ys4):

    label = "{:.0f}".format(x)

    ax4.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')     
    
for x,y in zip(xss4,ys4):

    label = "{:.1f}".format(x)

    ax4.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')     
    
    
for x,y in zip(xs5,ys5):

    label = "{:.1f}".format(x)

    ax5.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')     

for x,y in zip(xss5,ys5):

    label = "{:.1f}".format(x)

    ax5.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,-5),
                 ha='center')    
    
    
for x,y in zip(xs6,ys6):

    label = "{:.1f}".format(x)

    ax6.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss6,ys6):

    label = "{:.1f}".format(x)

    ax6.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')      
    
for x,y in zip(xs7,ys7):

    label = "{:.1f}".format(x)

    ax7.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss7,ys7):

    label = "{:.1f}".format(x)

    ax7.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,-5),
                 ha='center')  
    
for x,y in zip(xs8,ys8):

    label = "{:.1f}".format(x)

    ax8.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss8,yss8):

    label = "{:.1f}".format(x)
    

    ax8.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')    
    
for x,y in zip(xs9,ys9):

    label = "{:.1f}".format(x)

    ax9.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss9,ys9):

    label = "{:.1f}".format(x)

    ax9.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')      
    
    

for x,y in zip(xs10,ys10):

    label = "{:.0f}".format(x)

    ax10.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 
    

for x,y in zip(xss10,ys10):

    label = "{:.1f}".format(x)

    ax10.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 

for x,y in zip(xs11,ys11):

    label = "{:.1f}".format(x)

    ax11.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss11,ys11):

    label = "{:.1f}".format(x)

    ax11.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  
    
    
for x,y in zip(xs12,ys12):

    label = "{:.1f}".format(x)

    ax12.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss12,ys12):

    label = "{:.1f}".format(x)

    ax12.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  


# In[299]:


xss8


# In[300]:


DF7


# In[301]:


#расчет среднего по каждому показателю
#!!!!перевести все значения в оригинальной таблице в float!!!!!!, поправить код

DF7['1.7'] = (DF7['1.7'].astype(float)).mean()
DF7['2.1'] = (DF7['2.1'].str.replace(',','.').astype(float)).mean()
DF7['2.2'] = (DF7['2.2'].str.replace(',','.').astype(float)).mean()
DF7['2.3'] = (DF7['2.3'].str.replace(',','.').astype(float)).mean()
DF7['2.4'] = (DF7['2.4'].str.replace(',','.').astype(float)).mean()
DF7['2.5'] = (DF7['2.5'].str.replace(',','.').astype(float)).mean()
DF7['2.7'] = (DF7['2.7'].str.replace(',','.').astype(float)).mean()
#DF6['3.1'] = (DF6['3.1'].str.replace(',','.').astype(float)).mean
#DF6['3.3'] = (DF6['3.3'].str.replace(',','.').astype(float)).mean
DF7['6.2'] = (DF7['6.2'].str.replace(',','.').astype(float)).mean()
DF7['6.3'] = (DF7['6.3'].astype(float)).mean()
DF7['6.5'] = (DF7['6.5'].astype(float)).mean()

xxss = DF7['1.7']
xxss2 = DF7['2.1']
xxss3 = DF7['2.2']
xxss4 = DF7['2.3']
xxss5 = DF7['2.4']
xxss6 = DF7['2.5']
xxss7 = DF7['2.7']
#xxss8 = DF7['3.1']
#xxss9 = DF7['3.3']
xxss10 = DF7['6.2']
xxss11 = DF7['6.3']
xxss12 = DF7['6.5']


# In[303]:


xs = DF1.index
xs2 = DF1.index
xs3 = DF1.index
xs4 = DF1.index
xs5 = DF1.index
xs6 = DF1.index
xs7 = DF1.index
xs8 = DF1.index
xs9 = DF1.index
xs10 = DF1.index
xs11 = DF1.index
xs12 = DF1.index

xss = DF1.index
xss2 = DF1.index
xss3 = DF1.index
xss4 = DF1.index
xss5 = DF1.index
xss6 = DF1.index
xss7 = DF1.index
xss8 = DF1.index
xss9 = DF1.index
xss10 = DF1.index
xss11 = DF1.index
xss12 = DF1.index


ys = DF1['1.7']
ys2 = DF1['2.1']
ys3 = DF1['2.2']
ys4 = DF1['2.3']
ys5 = DF1['2.4']
ys6 = DF1['2.5']
ys7 = DF1['2.7']
ys8 = DF1['3.1']
ys9 = DF1['3.3']
ys10 = DF1['6.2']
ys11 = DF1['6.3']
ys12 = DF1['6.5']

yss = DF6['1.7']
yss2 = DF6['2.1']
yss3 = DF6['2.2']
yss4 = DF6['2.3']
yss5 = DF6['2.4']
yss6 = DF6['2.5']
yss7 = DF6['2.7']
yss8 = DF6['3.1']
yss9 = DF6['3.3']
yss10 = DF6['6.2']
yss11 = DF6['6.3']
yss12 = DF6['6.5']


xs, ys = zip(*sorted(zip(ys, xs)))
xs2, ys2 = zip(*sorted(zip(ys2, xs2)))
xs3, ys3 = zip(*sorted(zip(ys3, xs3)))
xs4, ys4 = zip(*sorted(zip(ys4, xs4)))
xs5, ys5 = zip(*sorted(zip(ys5, xs5)))
xs6, ys6 = zip(*sorted(zip(ys6, xs6)))
xs7, ys7 = zip(*sorted(zip(ys7, xs7)))
xs8, ys8 = zip(*sorted(zip(ys8, xs8)))
xs9, ys9 = zip(*sorted(zip(ys9, xs9)))
xs10, ys10 = zip(*sorted(zip(ys10, xs10)))
xs11, ys11 = zip(*sorted(zip(ys11, xs11)))
xs12, ys12 = zip(*sorted(zip(ys12, xs12)))

xss, yss = zip(*sorted(zip(yss,xss)))
xss2, yss2 = zip(*sorted(zip(yss2,xss2)))
xss3, yss3 = zip(*sorted(zip(yss3,xss3)))
xss4, yss4 = zip(*sorted(zip(yss4,xss4)))
xss5, yss5 = zip(*sorted(zip(yss5,xss5)))
xss6, yss6 = zip(*sorted(zip(yss6,xss6)))
xss7, yss7 = zip(*sorted(zip(yss7,xss7)))
xss8, yss8 = zip(*sorted(zip(yss8,xss8)))
xss9, yss9 = zip(*sorted(zip(yss9,xss9)))
xss10, yss10 = zip(*sorted(zip(yss10,xss10)))
xss11, yss11 = zip(*sorted(zip(yss11,xss11)))
xss12, yss12 = zip(*sorted(zip(yss12,xss12)))

fig, ((ax1, ax2, ax3, ax4), (ax5,ax6,ax7,ax8), (ax9,ax10, ax11,ax12) ) = plt.subplots(3, 4, figsize=(20,17)) 
fig.suptitle('Рейтинг институтов БФУ Июнь 2020',  fontsize=20) 

ax1.bar(ys,xs)
ax1.plot(ys,xs, marker='o')
ax1.plot(ys,xxss, c='gold')
ax1.bar(ys,xss)
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('Доля преподавателей, использующих ЭОР', fontsize=12)
ax1.text(1, 5, r'ЦелПоказ = 100%')

ax2.bar(ys2,xs2)
ax2.plot(ys2,xs2, marker='o')
#ax2.plot(ys,xxss2, c='gold')
ax2.bar(ys2,xss2)
ax2.tick_params(axis='x', labelrotation=90)
ax2.set_ylabel('Публикаций на 1 НПР (ядро РИНЦ)', fontsize=12)
ax2.text(1, 10, r'ЦелПоказ = 1')

ax3.bar(ys3,xs3)
ax3.plot(ys3,xs3, marker='o')
#ax3.plot(ys,xxss3, c='gold')
ax3.bar(ys3,xss3)
ax3.tick_params(axis='x', labelrotation=90)
ax3.set_ylabel('Публикаций на 1 НПР (Scopus и / или Web of Science)', fontsize=12)
ax3.text(1, 10, r'ЦелПоказ = 0,8')

ax4.bar(ys4,xs4)
ax4.plot(ys4,xs4, marker='o')
#ax4.plot(ys,xxss4, c='gold')
ax4.bar(ys4,xss4)
ax4.tick_params(axis='x', labelrotation=90)
ax4.set_ylabel('Доля НПР в I и II квартилях Scopus', fontsize=12)
ax4.text(1, 10, r'ЦелПоказ = 0,25')


ax5.bar(ys5,xs5)
ax5.plot(ys5,xs5, marker='o')
#ax5.plot(ys,xxss5, c='gold')
ax5.bar(ys5,xss5)
ax5.tick_params(axis='x', labelrotation=90)
ax5.set_ylabel('Цитируемость на 1 НПР (Scopus)', fontsize=12)
ax5.text(1, 10, r'ЦелПоказ = 4')

ax6.bar(ys6,xs6)
ax6.plot(ys6,xs6, marker='o')
#ax6.plot(ys,xxss6, c='gold')
ax6.bar(ys6,xss6)
ax6.tick_params(axis='x', labelrotation=90)
ax6.set_ylabel('Число заявок на НИОКР на 1 НПР', fontsize=12)
ax6.text(1, 10, r'ЦелПоказ = 0,053')

ax7.bar(ys7,xs7)
ax7.plot(ys7,xs7, marker='o')
ax7.plot(ys,xxss7, c='gold')
ax7.bar(ys7,xss7)
ax7.tick_params(axis='x', labelrotation=90)
ax7.set_ylabel('Объем договоров РИД на 1 НПР', fontsize=12)
ax7.text(1, 10, r'ЦелПоказ = 5,3')

ax8.bar(ys8,xs8)
ax8.plot(ys8,xs8, marker='o')
#ax8.plot(ys,xxss8, c='gold')
#ax8.bar(ys8,xss8) 
ax8.tick_params(axis='x', labelrotation=90)
ax8.set_ylabel('Объем НИОКР на 1 НПР', fontsize=12)
ax8.text(1, 5, r'ЦелПоказ = 900')

ax9.bar(ys9,xs9)
ax9.plot(ys9,xs9, marker='o')
#ax9.plot(ys,xxss9, c='gold')
#ax9.bar(ys9,xss9)
ax9.tick_params(axis='x', labelrotation=90)
ax9.set_ylabel('Объем внешних НИОКР на 1 НПР', fontsize=12)
ax9.text(1, 5, r'ЦелПоказ = 178')

ax10.bar(ys10,xs10)
ax10.plot(ys10,xs10, marker='o')
ax10.plot(ys,xxss10, c='gold')
ax10.bar(ys10,xss10)
ax10.tick_params(axis='x', labelrotation=90)
ax10.set_ylabel('Доля остепененных НПР', fontsize=12)
ax10.text(1, 5, r'ЦелПоказ = 80%')


ax11.bar(ys11,xs11)
ax11.plot(ys11,xs11, marker='o')
ax11.plot(ys,xxss11, c='gold')
ax11.bar(ys11,xss11)
ax11.tick_params(axis='x', labelrotation=90)
ax11.set_ylabel('Доля НПР до 35 лет', fontsize=12)
ax11.text(1, 5, r'ЦелПоказ = 30%')

ax12.bar(ys12,xs12)
ax12.plot(ys12,xs12, marker='o')
ax12.plot(ys,xxss12, c='gold')
ax12.bar(ys12,xss12)
ax12.tick_params(axis='x', labelrotation=90)
ax12.set_ylabel('Доля защитившихся аспирантов', fontsize=12)
ax12.text(1, 5, r'ЦелПоказ = 40%')

for x,y in zip(xs,ys):

    label = "{:.0f}".format(x)

    ax1.annotate(label, # this is the text
                 (y,x), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

for x,y in zip(xss,ys):

    label = "{:.1f}".format(x)

    ax1.annotate(label, # this is the text
                 (y,x), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
for x,y in zip(xs2,ys2):

    label = "{:.1f}".format(x)

    ax2.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 

for x,y in zip(xss2,ys2):

    label = "{:.1f}".format(x)

    ax2.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 

for x,y in zip(xs3,ys3):

    label = "{:.0f}".format(x)

    ax3.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 

for x,y in zip(xss3,ys3):

    label = "{:.1f}".format(x)

    ax3.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 
    
for x,y in zip(xs4,ys4):

    label = "{:.0f}".format(x)

    ax4.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')     

for x,y in zip(xss4,ys4):

    label = "{:.1f}".format(x)

    ax4.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 
    
for x,y in zip(xs5,ys5):

    label = "{:.0f}".format(x)

    ax5.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')     

for x,y in zip(xss5,ys5):

    label = "{:.1f}".format(x)

    ax5.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 
    
for x,y in zip(xs6,ys6):

    label = "{:.0f}".format(x)

    ax6.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss6,ys6):

    label = "{:.2f}".format(x)

    ax6.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,10),
                 ha='center',fontsize=8,fontweight='bold') 
    
for x,y in zip(xs7,ys7):

    label = "{:.0f}".format(x)

    ax7.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss7,ys7):

    label = "{:.1f}".format(x)

    ax7.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 
    
for x,y in zip(xs8,ys8):

    label = "{:.0f}".format(x)

    ax8.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  
    
for x,y in zip(xs9,ys9):

    label = "{:.0f}".format(x)

    ax9.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  



for x,y in zip(xs10,ys10):

    label = "{:.0f}".format(x)

    ax10.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 
    
for x,y in zip(xss10,ys10):

    label = "{:.1f}".format(x)

    ax10.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center') 

for x,y in zip(xs11,ys11):

    label = "{:.1f}".format(x)

    ax11.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss11,ys11):

    label = "{:.1f}".format(x)

    ax11.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  
    
    
for x,y in zip(xs12,ys12):

    label = "{:.1f}".format(x)

    ax12.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  

for x,y in zip(xss12,ys12):

    label = "{:.1f}".format(x)

    ax12.annotate(label, 
                 (y,x), 
                 textcoords="offset points", 
                 xytext=(-2,5),
                 ha='center')  


# In[304]:



DF = DF.rename(columns={'1.1': '% контракт. (13)',
                       '1.2': '% магистров и аспир. (14)',
                       '1.4': 'ЕГЭ, бюджет (15)',
                       '1.5': '% маг. и аспир. из др. вузов (16)',
                       '1.7': '% преп., исп. ЭОР (БРС) (12)',
                       '1.9': '% программ с онлайн (18)',
                       '1.10': '% общ. занятости вып. (19)',
                       '3.5': '% Задолж. (20)',
                       '4.1': '% иностранных студентов (21)',
                       '6.2': '% остепененных НПР (10)',
                       '6.3': '% НПР до 35 лет (11)',
                       '1.8': '% совместных программ (17)'})

DF3 = DF3.rename(columns={'1.1': '% контракт. (13)',
                       '1.2': '% магистров и аспир. (14)',
                       '1.4': 'ЕГЭ, бюджет (15)',
                       '1.5': '% маг. и аспир. из др. вузов (16)',
                       '1.7': '% преп., исп. ЭОР (БРС) (12)',
                       '1.9': '% программ с онлайн (18)',
                       '1.10': '% общ. занятости вып. (19)',
                       '3.5': '% Задолж. (20)',
                       '4.1': '% иностранных студентов (21)',
                       '6.2': '% остепененных НПР (10)',
                       '6.3': '% НПР до 35 лет (11)',
                       '1.8': '% совместных программ (17)'})


# In[305]:


DF["sum"] = DF.round(2).sum(axis=1)
DF["sum"] = DF["sum"]/12                
                
DF11 = DF["sum"].sort_values(ascending=True).plot.line()
axes44 = DF["sum"].round(1).sort_values(ascending=True).plot.bar(title = 'Итог декабрь',figsize=(10,7))

totals4 = []
for r in axes44.patches:
    totals4.append(r.get_height())
total4 = sum(totals4)
for r in axes44.patches:  
    axes44.text(r.get_x(), r.get_height(),             r.get_height(), fontsize=12,
                color='black')
    
import seaborn as sns

cm = sns.light_palette("green", as_cmap=True)

ddf1 = DF.T.round(2).style.background_gradient(cmap=cm)
#ddf2 = DF3.T.round(2).style.background_gradient(cmap=cm)


display(DF3.T.round(2).style.background_gradient(cmap=cm))
display(ddf1)

#display(ddf2)


# In[306]:


DF1 = DF1.rename(columns={'1.7': '% преп., исп. ЭОР (БРС) (12)',
                       '2.1': 'Публикаций на 1 НПР (ядро РИНЦ) (1)',
                       '2.2': 'Публикаций на 1 НПР (Scopus и / или Web of Science) (2)',
                       '2.3': 'Доля НПР с публикациями в I и II квартилях Scopus (3)',
                       '2.4': 'Цитируемость на 1 НПР (Scopus) (4)',
                       '2.5': 'Число заявок на НИОКР на 1 НПР (5)',
                       '2.7': 'Объем договоров РИД на 1 НПР (6)',
                       '3.1': 'Объем НИОКР на 1 НПР (7)',
                       '3.3': 'Объем внешних НИОКР на 1 НПР (8)',
                       '6.2': '% остепененных НПР (10)',
                       '6.3': '% НПР до 35 лет (11)',
                       '6.5': '% защитившихся аспирантов (9)'})


DF4 = DF4.rename(columns={'1.7': '% преп., исп. ЭОР (БРС) (12)',
                       '2.1': 'Публикаций на 1 НПР (ядро РИНЦ) (1)',
                       '2.2': 'Публикаций на 1 НПР (Scopus и / или Web of Science) (2)',
                       '2.3': 'Доля НПР с публикациями в I и II квартилях Scopus (3)',
                       '2.4': 'Цитируемость на 1 НПР (Scopus) (4)',
                       '2.5': 'Число заявок на НИОКР на 1 НПР (5)',
                       '2.7': 'Объем договоров РИД на 1 НПР (6)',
                       '3.1': 'Объем НИОКР на 1 НПР (7)',
                       '3.3': 'Объем внешних НИОКР на 1 НПР (8)',
                       '6.2': '% остепененных НПР (10)',
                       '6.3': '% НПР до 35 лет (11)',
                       '6.5': '% защитившихся аспирантов (9)'})



# In[307]:


DF1["sum"] = DF1.round(2).sum(axis=1)
DF1["sum"] = DF1["sum"]/12                
                
DF11 = DF1["sum"].sort_values(ascending=True).plot.line()
axes44 = DF1["sum"].round(1).sort_values(ascending=True).plot.bar(title = 'Итог июнь',figsize=(10,7))

totals4 = []
for r in axes44.patches:
    totals4.append(r.get_height())
total4 = sum(totals4)
for r in axes44.patches:  
    axes44.text(r.get_x(), r.get_height(),             r.get_height(), fontsize=12,
                color='black')
    
import seaborn as sns

cm = sns.light_palette("green", as_cmap=True)

ddf1 = DF1.T.round(2).style.background_gradient(cmap=cm)
#ddf2 = DF3.T.round(2).style.background_gradient(cmap=cm)


display(DF4.T.round(2).style.background_gradient(cmap=cm))
display(ddf1)

#display(ddf2)


# In[308]:


axes444 = DF3.plot.bar(subplots=False, figsize=(24,15))

totals4 = []
for r in axes444.patches:
    totals4.append(r.get_height())
total4 = sum(totals4)
for r in axes444.patches:  
    axes444.text(r.get_x(), r.get_height(),             r.get_height(), fontsize=12,
                color='black')


# In[309]:


DF6 = DF6.rename(columns={'1.7': '% преп., исп. ЭОР (БРС) (12)',
                       '2.1': 'Публикаций на 1 НПР (ядро РИНЦ) (1)',
                       '2.2': 'Публикаций на 1 НПР (Scopus и / или Web of Science) (2)',
                       '2.3': 'Доля НПР с публикациями в I и II квартилях Scopus (3)',
                       '2.4': 'Цитируемость на 1 НПР (Scopus) (4)',
                       '2.5': 'Число заявок на НИОКР на 1 НПР (5)',
                       '2.7': 'Объем договоров РИД на 1 НПР (6)',
                       '3.1': 'Объем НИОКР на 1 НПР (7)',
                       '3.3': 'Объем внешних НИОКР на 1 НПР (8)',
                       '6.2': '% остепененных НПР (10)',
                       '6.3': '% НПР до 35 лет (11)',
                       '6.5': '% защитившихся аспирантов (9)'})

DF6 = DF6.rename(index={'Инженерно-технический институт': 'Инж-тех',
                 'Институт гуманитарных наук': 'ИГН',
                 'Институт живых систем': 'ИЖС',
                 'Институт образования': 'ИнОбр.',
                 'Институт природопользования территориального развития и градостроительства': 'ИПТРиГ',
                 'Институт рекреации, туризма и физической культуры': 'ИРТиФК',
                 'Институт физико-математических наук и информационных технологий': 'Физ-мат',
                 'Институт экономики и менеджмента': 'ИЭиМ',
                 'Медицинский институт': 'МедИн',
                 'Юридический институт': 'ЮридИн'})

axes444 = DF6.plot.bar(subplots=False, figsize=(24,15), ylim=(0,300))

totals4 = []
for r in axes444.patches:
    totals4.append(r.get_height())
total4 = sum(totals4)
for r in axes444.patches:
    axes444.text(r.get_x(), r.get_height(),             r.get_height(), fontsize=12,
                color='black')

