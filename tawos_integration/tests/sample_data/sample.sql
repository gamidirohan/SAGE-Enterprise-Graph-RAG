-- Sample SQL file for testing

CREATE TABLE `documents` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(255) NOT NULL,
  `content` text NOT NULL,
  `type` varchar(50) DEFAULT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
);

CREATE TABLE `entities` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `type` varchar(50) NOT NULL,
  `document_id` int(11) NOT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `document_id` (`document_id`),
  CONSTRAINT `entities_ibfk_1` FOREIGN KEY (`document_id`) REFERENCES `documents` (`id`)
);

CREATE TABLE `relationships` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `source_id` int(11) NOT NULL,
  `target_id` int(11) NOT NULL,
  `type` varchar(50) NOT NULL,
  `document_id` int(11) NOT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `source_id` (`source_id`),
  KEY `target_id` (`target_id`),
  KEY `document_id` (`document_id`),
  CONSTRAINT `relationships_ibfk_1` FOREIGN KEY (`source_id`) REFERENCES `entities` (`id`),
  CONSTRAINT `relationships_ibfk_2` FOREIGN KEY (`target_id`) REFERENCES `entities` (`id`),
  CONSTRAINT `relationships_ibfk_3` FOREIGN KEY (`document_id`) REFERENCES `documents` (`id`)
);

-- Sample data
INSERT INTO `documents` (`id`, `title`, `content`, `type`) VALUES
(1, 'Sample Document 1', 'This is a sample document for testing.', 'text'),
(2, 'Sample Document 2', 'Another sample document with different content.', 'text'),
(3, 'Technical Report', 'This technical report discusses various aspects of AI.', 'report');

INSERT INTO `entities` (`id`, `name`, `type`, `document_id`) VALUES
(1, 'sample', 'keyword', 1),
(2, 'document', 'keyword', 1),
(3, 'testing', 'keyword', 1),
(4, 'sample', 'keyword', 2),
(5, 'document', 'keyword', 2),
(6, 'content', 'keyword', 2),
(7, 'technical', 'keyword', 3),
(8, 'report', 'keyword', 3),
(9, 'AI', 'concept', 3);

INSERT INTO `relationships` (`id`, `source_id`, `target_id`, `type`, `document_id`) VALUES
(1, 1, 2, 'RELATED_TO', 1),
(2, 2, 3, 'RELATED_TO', 1),
(3, 4, 5, 'RELATED_TO', 2),
(4, 5, 6, 'RELATED_TO', 2),
(5, 7, 8, 'RELATED_TO', 3),
(6, 8, 9, 'RELATED_TO', 3);
