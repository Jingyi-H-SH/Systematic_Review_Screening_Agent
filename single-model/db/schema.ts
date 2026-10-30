import { sqliteTable, text, integer } from 'drizzle-orm/sqlite-core';
export const projects = sqliteTable('projects', {
  id: text('id').primaryKey(), title: text('title').notNull(), criteria: text('criteria').notNull(),
  constructs: text('constructs').notNull(), created: text('created').notNull(),
  reviewType: text('review_type').notNull().default('general'),
  ownerId: text('owner_id').notNull().default('local-development'),
  extractionTemplate: text('extraction_template').notNull().default('Study design; country and setting; population; sample size; intervention or exposure; comparator; outcomes; follow-up; effect estimates with uncertainty; funding; limitations'),
});
export const searches = sqliteTable('searches', {
  id: text('id').primaryKey(), projectId: text('project_id').notNull(), database: text('database').notNull(),
  query: text('query').notNull(), searched: text('searched').notNull(), count: integer('count').notNull(), created: text('created').notNull(),
});
export const studies = sqliteTable('studies', {
  id: text('id').primaryKey(), projectId: text('project_id').notNull(), title: text('title').notNull(),
  abstract: text('abstract').notNull(), doi: text('doi').notNull(), document: text('document'),
  created: text('created').notNull(),
});
export const jobs = sqliteTable('jobs', {
  id: text('id').primaryKey(), studyId: text('study_id').notNull(), kind: text('kind').notNull(),
  status: text('status').notNull(), phase: integer('phase').notNull(), lease: text('lease'),
  data: text('data').notNull(), error: text('error'), created: text('created').notNull(), updated: text('updated').notNull(),
});
export const decisions = sqliteTable('decisions', {
  id: text('id').primaryKey(), studyId: text('study_id').notNull(), jobId: text('job_id'),
  kind: text('kind').notNull(), actor: text('actor').notNull(), data: text('data').notNull(), created: text('created').notNull(),
});
