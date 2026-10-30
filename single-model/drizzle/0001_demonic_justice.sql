CREATE TABLE `searches` (
	`id` text PRIMARY KEY NOT NULL,
	`project_id` text NOT NULL,
	`database` text NOT NULL,
	`query` text NOT NULL,
	`searched` text NOT NULL,
	`count` integer NOT NULL,
	`created` text NOT NULL
);
--> statement-breakpoint
ALTER TABLE `projects` ADD `review_type` text DEFAULT 'general' NOT NULL;--> statement-breakpoint
ALTER TABLE `projects` ADD `extraction_template` text DEFAULT 'Study design; country and setting; population; sample size; intervention or exposure; comparator; outcomes; follow-up; effect estimates with uncertainty; funding; limitations' NOT NULL;