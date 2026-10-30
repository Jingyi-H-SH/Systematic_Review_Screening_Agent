CREATE TABLE `decisions` (
	`id` text PRIMARY KEY NOT NULL,
	`study_id` text NOT NULL,
	`job_id` text,
	`kind` text NOT NULL,
	`actor` text NOT NULL,
	`data` text NOT NULL,
	`created` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `jobs` (
	`id` text PRIMARY KEY NOT NULL,
	`study_id` text NOT NULL,
	`kind` text NOT NULL,
	`status` text NOT NULL,
	`phase` integer NOT NULL,
	`lease` text,
	`data` text NOT NULL,
	`error` text,
	`created` text NOT NULL,
	`updated` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `projects` (
	`id` text PRIMARY KEY NOT NULL,
	`title` text NOT NULL,
	`criteria` text NOT NULL,
	`constructs` text NOT NULL,
	`created` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `studies` (
	`id` text PRIMARY KEY NOT NULL,
	`project_id` text NOT NULL,
	`title` text NOT NULL,
	`abstract` text NOT NULL,
	`doi` text NOT NULL,
	`document` text,
	`created` text NOT NULL
);
