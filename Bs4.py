import pymysql
import urllib
from bs4 import BeautifulSoup
import winsound

def dbCreateTable():
    sqlCreateTable = "CREATE TABLE vestelvpress2000 (urun_adi TEXT(3000),oylama INT,yorum TEXT(3000))"
    mycursor.execute(sqlCreateTable)
    db.commit()

def dbDelete():
    sql2 = "DELETE FROM samsung-galaxy-j7-prime"
    sql3 = "ALTER TABLE samsung-galaxy-j7-prime AUTO_INCREMENT = 1"
    mycursor.execute(sql2)
    mycursor.execute(sql3)

def dbWrite(productname1, rating1, comment):
    sql = "INSERT INTO vestelvpress2000(urun_adi, oylama, yorum) VALUES (%s, %s, %s)"
    val = (productname1,  rating1, comment)
    mycursor.execute(sql, val)
    db.commit()


# Open database connection
db = pymysql.connect("localhost", USERNAME, PASSWORD, TABLENAME)
# prepare a cursor object using cursor() method
mycursor = db.cursor()

dbCreateTable()
kategori="cep telefonu"
gtUrl1 = "https://www.gittigidiyor.com/elektrikli-ev-aletleri/utu-utu-masasi/utu/vestel-v-press-2000-buhar-jeneratorlu/yorumlari?sf={}"
gtUrl=[]
gtUrl.append(gtUrl1)
comment1=''

for l in gtUrl:
    theUrl = l
    thePage = urllib.urlopen(theUrl)
    soup = BeautifulSoup(thePage, "html.parser")
    s = soup.find('ul', {"class": "clearfix"})
    s1 = soup.find('span', {"class": "catalog-review-link-small"})
    urlS = ''.join(x for x in s1.text if x.isdigit())
    urlSayisi = int(urlS)
    urlSayisi = urlSayisi / 10 + 2

    for k in range(1,urlSayisi):
        theUrl1 = l.format(k)
        thePage = urllib.urlopen(theUrl1)
        soup = BeautifulSoup(thePage, "html.parser")
        print(theUrl1)


        for s in soup.findAll('div', {"class": "catalog-review-title clearfix"}):
            n = s.find('div', {"class": "gg-d-24 catalog-name"})
            productName = n.text

        s = soup.find('div', {"class": "gg-d-24 padding-none"})
        for s in soup.findAll('div', {"class": "user-catalog-review clearfix"}):

            n = s.find('div', {"class": "gg-d-24 gg-t-24 gg-m-20 padding-none user-detail-container"})
            username=n.a.get('title')
            tarih2 = n.span.text


            try:
                r = s.find('div', {"class": "left"})
                r1 = r.text
            except AttributeError:
                print ("rating yok")
            try:
                t = s.find('h3')
                title = t.text
            except AttributeError:
                print("comment title doesn'texist")

            for c in s.findAll('div', {"class": "user-catalog-review-comment-detail gg-d-23 gg-m-22 pl0"}):
                comment1 = c.text


            if (comment1!=""):
                try:
                    dbWrite(productName.encode('utf-8'), r1, comment1.encode('utf-8'))
                except pymysql.err.InternalError:
                    dbWrite("null", 0, "null")



            comment1=""
            r1=0
            title=""

# disconnect from server
db.close()
winsound.Beep(2500,5000)