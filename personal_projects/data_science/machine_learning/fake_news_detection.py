
#Import necessary modules
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#Read CSV data file
df = pd.read_csv('~/personal_projects/data_science/data/news.csv')

df.shape
df.head()

#Obtain headers of columns of CSV data file
labels = df.label
labels.head()

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')
perc_err = 100.0 - round(score * 100, 2)

confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

#Identify where predicted property is different than test set
real_error_count = np.where((y_test.values=='REAL') & (y_pred == 'FAKE'))[0]
fake_error_count = np.where((y_test.values=='FAKE') & (y_pred == 'REAL'))[0]

#Identify where predicted property is the same as the test set
real_true_count = np.where((y_test.values=='REAL') & (y_pred == 'REAL'))[0]
fake_true_count = np.where((y_test.values=='FAKE') & (y_pred == 'FAKE'))[0]

#Create array identifying which data were correctly ('Correct') or incorrectly
# ('Error') identified by algorithm
error_arr = np.empty(len(y_test.values), dtype=str)

error_arr[real_error_count] = 'Error'
error_arr[fake_error_count] = 'Error'
error_arr[real_true_count] = 'Correct'
error_arr[fake_true_count] = 'Correct'

#Create data frame snapshot describing notable data
data_class = pd.DataFrame({'init': y_test, 'Prediction': y_pred, 'Accuracy':error_arr})

#Define plot parameters
fsize=10
y_step = 100
green = ("#77dd77")
red = ("#ff6961")
plot_min = 0

if (len(real_error_count)>=len(fake_error_count)) & (len(real_error_count)>=len(real_true_count)) & (len(real_error_count)>=len(fake_true_count)):
    temp_max = len(real_error_count)
elif (len(fake_error_count)>=len(real_error_count)) & (len(fake_error_count)>=len(real_true_count)) & (len(fake_error_count)>=len(fake_true_count)):
    temp_max = len(fake_error_count)
elif (len(real_true_count)>=len(fake_error_count)) & (len(real_true_count)>=len(real_error_count)) & (len(real_true_count)>=len(fake_true_count)):
    temp_max = len(real_true_count)
elif (len(fake_true_count)>=len(fake_error_count)) & (len(fake_true_count)>=len(real_error_count)) & (len(fake_true_count)>=len(real_true_count)):
    temp_max = len(fake_true_count)

plot_max = temp_max + 100
if (plot_max%50==0):
    plot_max += 1
    while (plot_max%50!=0):
        plot_max += 1
else:
    while (plot_max%50!=0):
        plot_max += 1

text_pos = (temp_max / plot_max) + (plot_max - temp_max)/(2*plot_max)

#Bar graph showing effectiveness of fake news predictor
fig, ax1 = plt.subplots(figsize=(8,6))

color_arr = {'E': red, 'C': green}
sns.set_theme(style="darkgrid")
sns.countplot(x='Prediction', hue="Accuracy", data=data_class, palette=color_arr, linewidth=2, edgecolor='black')

#Define x,y axes
plt.ylabel('Counts [N]', fontsize=2*fsize)
plt.xlabel('Prediction', fontsize=2*fsize)

plt.ylim([plot_min, plot_max])

# Set x,y tick parameters
loc = ticker.MultipleLocator(base=y_step)
ax1.yaxis.set_major_locator(loc)
loc = ticker.MultipleLocator(base=round(y_step/2,0))
ax1.yaxis.set_minor_locator(loc)

ax1.tick_params(axis='x', labelsize=fsize*1.5)
ax1.tick_params(axis='y', labelsize=fsize*1.5)

#Define legends/text
plt.text(0.03, text_pos, s=r'% Error = ' + '{:.2f}'.format(perc_err) + ' ' + r'%',
    fontsize=fsize*2.0,transform = ax1.transAxes,
    bbox=dict(facecolor=(1,1,1,0.75), edgecolor=(0,0,0,1), boxstyle='round, pad=0.1'))

#Output plot
plt.tight_layout()
plt.show()
#plt.savefig('~/personal_projects/data_science/outputs/fake_news_histogram.png', dpi=150)
#plt.close()
