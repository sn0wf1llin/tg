__author__ = 'MA573RWARR10R'
from peewee_classes import *


def truncate_topics_tables(resource):
    database.execute_sql('SET FOREIGN_KEY_CHECKS = 0')

    t_r_ids = Topics.select(Topics.topic).join(TopicsResources, on=(TopicsResources.topic == Topics.topic)).where(
        TopicsResources.resource == resource).execute()

    for t in t_r_ids:
        dt = Topics.get(topic=t.get().__data__['topic'])
        dt.delete_instance()

    TopicsResources.delete().where(TopicsResources.resource == resource).execute()

    database.execute_sql('SET FOREIGN_KEY_CHECKS = 1')
