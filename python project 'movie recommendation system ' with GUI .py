#!E:\3rd sem\python project\ml-100k
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings # package in python
warnings.filterwarnings('ignore')


# # GET THE DATA SET 

# In[2]:


columns_names = ["user_id","item_id","rating","timestamp"]
df = pd.read_csv("ml-100k/u.data",sep='\t',names = columns_names)
df.head() 


# In[3]:



df.shape #means 1 lakh rows and 4 col.


# In[4]:


##columns_names = ["user_id","iteam_name","rating","timestamp"] -- giving columns names after reading the data


# In[5]:


##df = pd.read_csv("ml-100k/u.data",sep ='\t',names = columns_names)- alloting to table 


# In[6]:


df['user_id'].nunique() ## unnique users who rated the movies (according to collected data)


# In[7]:


df['item_id'].nunique() ## NO of movies


# In[8]:


movies_titles = pd.read_csv("ml-100k/u.item",sep='\|',header = None)


# In[9]:


movies_titles.shape


# In[10]:


movies_titles = movies_titles[[0,1]]
movies_titles.columns=["item_id","title"]


# In[11]:


movies_titles.head()


# # NOW WE HAVE TWO DATA FRAME SO WE NEED TO COMBINE IT 

# In[12]:


df = pd.merge(df,movies_titles, on ="item_id")
df.head()
df.tail()


# # exploratory data analysis

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# In[14]:


df.groupby('title').mean()


# In[15]:


df.groupby('title').mean()['rating'].sort_values(ascending = False)


# In[16]:


df.groupby('title').count()


# In[17]:


# now we able to see how many rating a perticuler movie have


# In[18]:


df.groupby('title').count()['rating'].sort_values(ascending = False)


# ##most rated movie is star war as value is 583 then contact

# In[19]:


ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings.head()    


# In[20]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
ratings


# In[21]:


ratings.sort_values(by ='rating',ascending = False)


# # NOW WE SEE SOME GRAPH TO MORE RECOGNIZE THE RELATION BETWEEN MOVIES AND USER INTREACTION ON IT BY USING HISTOGRAM FUNCTION TO PLOT GRAPH

# In[22]:


plt.figure(figsize = (10 , 6))
plt.hist(ratings['num of ratings'], bins = 70)
plt.show()


# In[23]:


# here  x - num of ratings given by user 
       # y - num of time  


# In[24]:


plt.hist(ratings['rating'], bins = 70)
plt.show()


# In[25]:


# y - users who rated the movies
# x - av rating given 


# In[26]:


sns.jointplot(x='rating',y = 'num of ratings',data = ratings,alpha = 0.5)


# In[27]:


df.head()


# In[28]:


moviemat=df.pivot_table(index ="user_id", columns = "title",values = "rating")


# In[29]:


moviemat.head()


# In[30]:


## calling perticuler one 


# In[31]:


ratings.sort_values('num of ratings', ascending = False).head()


# In[32]:


starwars_user_ratings = moviemat['Star Wars (1977)']
starwars_user_ratings.head()


# # NOW MAIN PART WHICH TELLS THE CORELATION BTW THE MOVIES WE FIND 
#  ## THIS BY PANDAS FUNCTION CORELATION SIMPLY WITH RESPECT TO  OTHER MOVIES

# In[33]:


moviemat.corrwith(starwars_user_ratings)


# # rang of values from +1 to -1
# - 1 represents high relation 
# - -1 '''''' low relation 

# In[34]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[35]:


corr_starwars = pd.DataFrame(similar_to_starwars, columns = ['correlation']) 


# In[36]:


corr_starwars.dropna(inplace =True)
corr_starwars.head()


# # recommend high corelation movies

# In[37]:


corr_starwars.sort_values('correlation',ascending = False).head(10)


# # now after collecting the data of corelation i found an error if only on e two people had given rating then my recommendation will going to wrong so as a conculsion i applied filter on number of ratings 

# In[38]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[39]:


corr_starwars.head()


# In[40]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending = False)


# In[41]:


#from ipynb.fs.full.lusi import printValue


# In[ ]:





# In[42]:


from tkinter import *

ws = Tk()
ws.title("MOVIE RECOMMENDATION SYSTEM")
ws.geometry('400x300')
ws['bg'] = '#ffbf00'

def transfer(self):
    self.name = search_name.get() 
    return self.name

def printValue():
    pname = search_name.get()
    Label(ws, text=f'{pname}, Registered!', pady=20, bg='#ffbf00').pack()


search_name = Entry(ws)
search_name.pack(pady=30)

Button(
    ws,
    text="searched movie !", 
    padx=10, 
    pady=5,
    command=printValue
    ).pack()



#lst = data_dict['correlation']
#t = Text(ws)
#for x in lst:
    #t.insert(END, x + '\n')
#t.pack()


ws.mainloop()


# In[43]:


movie_name = input()
  


# # Predict the movie 

# In[44]:



def predict_movies(movie_name):
    
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns = ['correlation'])
    corr_movie.dropna(inplace =True)
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions =corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending = False)
   

    return predictions


# In[ ]:





# In[45]:


predictions =predict_movies(movie_name)
predictions.head()


# In[46]:



# Creating the index
idx = predictions.correlation

# Print the Index
print(idx)


# In[47]:


data_dict = predictions.to_dict()


# In[48]:


#data_dict


# In[49]:


#dict_file = data_dict
 
# Converting into list of tuple
#list = [(k, v) for k, v in dict.items()]
#print(dict)


# In[50]:


#from tkinter import * 
from tkinter import *

lst = data_dict['correlation']

root = Tk()
t = Text(root)
for x in lst:
    t.insert(END, x + '\n')
t.pack()
root.mainloop()


# In[ ]:





# In[ ]:




