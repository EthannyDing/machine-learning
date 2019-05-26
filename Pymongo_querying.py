# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:06:37 2019

@author: Ethan
"""

from pymongo import MongoClient
client = MongoClient('52.168.92.204', 27019)
db = client.test_database

collection = client.test_collection
posts = db.posts

import datetime
post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}
post_id = posts.insert_one(post).inserted_id
posts.find_one()
new_posts = [{"author": "Ethan",
               "text": "Another post!",
               "tags": ["bulk", "insert"],
               "date": datetime.datetime(2009, 11, 12, 11, 14)},
             {"author": "Eliot",
              "title": "MongoDB is fun",
              "text": "and pretty easy too!",
              "date": datetime.datetime(2009, 11, 10, 10, 45)}]

results = posts.insert_many(new_posts)
results.inserted_ids
posts.find_one({'author':'Eliot'})

'''
from bson.objectid import ObjectId
def get(post_id):
    # Convert from string to ObjectId:
    document = client.db.collection.find_one({'_id': ObjectId(post_id)})
    return document
doc = get(posts['posts._id'])
'''
db.list_collection_names()
client.list_database_names()
import pprint
pprint.pprint(posts.find_one({'author': 'Ethan'}))
pprint.pprint(posts.find({'author': 'Ethan'}))
for x in posts.find({'author': 'Ethan'}):
    print(x)

# querying for more than one document
for post in posts.find():
    pprint.pprint(post)

# counting  
posts.count_documents({})
posts.count_documents({'author': 'Mike'})
for title in posts.find({},{'_id':0,'text': 1}):
    print(title)

for title in posts.find().sort('author', -1):
    pprint.pprint(title)

posts.delete_one({'author': 'Eliot'})

# update documents
old = {'text':'My first blog post!'}
new = {'$set': {'text': 'This is my second post!'}}    
posts.update_one(old, new)

posts.drop({})

local = client.dev_owen
local.list_collection_names()[:20]

    