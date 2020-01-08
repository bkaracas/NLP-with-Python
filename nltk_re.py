import re
import pymysql
from nltk.corpus import stopwords


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('turkish'))
def printer(rat_cor_ar,com_cor_ar):
        f = open("j.csv", "w")
        i = 0
        f.write("post,tag\n")
        for i in range(0, len(rat_cor_ar)):
            if int(rat_cor_ar[i]) <= 3:
                f.write(com_cor_ar[i])
                f.write(",")
                f.write('neg')
                f.write("\n")
            else:
                f.write(com_cor_ar[i])
                f.write(",")
                f.write('pos')
                f.write("\n")


db = pymysql.connect("localhost", username, password, db)
# prepare a cursor object using cursor() method
mycursor = db.cursor()
mycursor1 = db.cursor()

sql2 = "SELECT yorum FROM xiaomiredmi5plus"
mycursor.execute(sql2)
sql1='select oylama from xiaomiredmi5plus'
mycursor1.execute(sql1)
com_db = mycursor.fetchall()
rat_db = mycursor1.fetchall()

rat_cor_ar=[]
com_cor_ar=[]
print rat_cor_ar
for comments in com_db:
    re.sub(r'[\x00-\x1f\x7f-\x9f]', " ", str(comments))

    comments = str(comments)
    comments = comments.encode("cp857")
    comments = comments.replace("('", "")
    comments = comments.replace("('", "")
    comments = comments.replace("',)", "")
    comments = comments.replace(".", " ")
    comments = comments.replace(",", " ")
    comments = comments.replace("!", "")
    comments = comments.replace("\\n", " ")
    comments = comments.replace("\\xc3\\xbc", "u")
    comments = comments.replace("\\xc3\\x9c", "u")
    comments = comments.replace("\\xc4\\xb1", "i")
    comments = comments.replace("\\xc4\\xb1", "i")
    comments = comments.replace("\\xc5\\x9f", "s")
    comments = comments.replace("\\xc5\\x9e", "s")
    comments = comments.replace("\\xc3\\xb6", "o")
    comments = comments.replace("\\xc4\\x9f", "g")
    comments = comments.replace("\\xc4\\x9e", "g")
    comments = comments.replace("\\xc3\\xa7", "c")
    comments = comments.replace("\\xc3\\x87", "c")
    comments = comments.replace("\\xc4\\xb0", "i")
    comments = comments.replace("\\xc2\\xa0", "")
    comments = comments.replace("\\xc3\\x96", "o")
    comments = comments.replace("\\xc3\\xb9", "u")
    comments = comments.replace("\\xe2\\x80\\x99", "'")
    comments = comments.replace("\\xe2\\x80\\x9c", "'")
    comments = comments.replace("\\xe2\\x80\\xa6", " ")
    comments = comments.replace("#", " ")
    comments = comments.replace("%", " ")
    comments = comments.replace("'", " ")
    comments = comments.replace(" '' ", " ")
    comments = comments.replace("(", " ")
    comments = re.sub(r'[^a-zA-Z]', " ", str(comments))
    comments = re.sub(r"\b[a-zA-Z]\b", " ", str(comments))
    comments = comments.lower()
    comments = REPLACE_BY_SPACE_RE.sub(' ', comments)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    comments = BAD_SYMBOLS_RE.sub('', comments)  # delete symbols which are in BAD_SYMBOLS_RE from text
    comments = ' '.join(word for word in comments.split() if word not in STOPWORDS)  # delete stopwors from text
    com_cor_ar.append(comments)
print rat_cor_ar
for comments in rat_db:
    re.sub(r'[\x00-\x1f\x7f-\x9f]', " ", str(comments))

    comments = str(comments)
    comments = comments.encode("cp857")
    comments = comments.replace("('", "")
    comments = comments.replace("('", "")
    comments = comments.replace("',)", "")
    comments = comments.replace(",)", "")
    comments = comments.replace("#", " ")
    comments = comments.replace("%", " ")
    comments = comments.replace("'", " ")
    comments = comments.replace(" '' ", " ")
    comments = comments.replace("(", " ")
    rat_cor_ar.append(comments)

print rat_cor_ar
print len(rat_cor_ar)
print len(com_cor_ar)
printer(rat_cor_ar,com_cor_ar)


