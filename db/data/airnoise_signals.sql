-- MySQL dump 10.13  Distrib 8.0.16, for Win64 (x86_64)
--
-- Host: localhost    Database: airnoise
-- ------------------------------------------------------
-- Server version	5.7.25-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
 SET NAMES utf8 ;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Dumping data for table `signals`
--

LOCK TABLES `signals` WRITE;
/*!40000 ALTER TABLE `signals` DISABLE KEYS */;
INSERT INTO `signals` (`measurement`, `array`, `microphone`) VALUES (1,1,1),(1,1,2),(1,1,3),(1,1,4),(1,2,1),(1,2,2),(1,2,3),(1,2,4),(1,3,1),(1,3,2),(1,3,3),(1,3,4),(2,1,1),(2,1,2),(2,1,3),(2,1,4),(2,2,1),(2,2,2),(2,2,3),(2,2,4),(2,3,1),(2,3,2),(2,3,3),(2,3,4),(3,1,1),(3,1,2),(3,1,3),(3,1,4),(3,2,1),(3,2,2),(3,2,3),(3,2,4),(3,3,1),(3,3,2),(3,3,3),(3,3,4),(4,1,1),(4,1,2),(4,1,3),(4,1,4),(4,2,1),(4,2,2),(4,2,3),(4,2,4),(4,3,1),(4,3,2),(4,3,3),(4,3,4),(5,1,1),(5,1,2),(5,1,3),(5,1,4),(5,2,1),(5,2,2),(5,2,3),(5,2,4),(5,3,1),(5,3,2),(5,3,3),(5,3,4),(6,1,1),(6,1,2),(6,1,3),(6,1,4),(6,2,1),(6,2,2),(6,2,3),(6,2,4),(6,3,1),(6,3,2),(6,3,3),(6,3,4),(7,1,1),(7,1,2),(7,1,3),(7,1,4),(7,2,1),(7,2,2),(7,2,3),(7,2,4),(7,3,1),(7,3,2),(7,3,3),(7,3,4),(8,1,1),(8,1,2),(8,1,3),(8,1,4),(8,2,1),(8,2,2),(8,2,3),(8,2,4),(8,3,1),(8,3,2),(8,3,3),(8,3,4),(9,1,1),(9,1,2),(9,1,3),(9,1,4),(9,2,1),(9,2,2),(9,2,3),(9,2,4),(9,3,1),(9,3,2),(9,3,3),(9,3,4),(10,1,1),(10,1,2),(10,1,3),(10,1,4),(10,2,1),(10,2,2),(10,2,3),(10,2,4),(10,3,1),(10,3,2),(10,3,3),(10,3,4),(11,1,1),(11,1,2),(11,1,3),(11,1,4),(11,2,1),(11,2,2),(11,2,3),(11,2,4),(11,3,1),(11,3,2),(11,3,3),(11,3,4),(12,1,1),(12,1,2),(12,1,3),(12,1,4),(12,2,1),(12,2,2),(12,2,3),(12,2,4),(12,3,1),(12,3,2),(12,3,3),(12,3,4),(13,1,1),(13,1,2),(13,1,3),(13,1,4),(13,2,1),(13,2,2),(13,2,3),(13,2,4),(13,3,1),(13,3,2),(13,3,3),(13,3,4),(14,1,1),(14,1,2),(14,1,3),(14,1,4),(14,2,1),(14,2,2),(14,2,3),(14,2,4),(14,3,1),(14,3,2),(14,3,3),(14,3,4),(15,1,1),(15,1,2),(15,1,3),(15,1,4),(15,2,1),(15,2,2),(15,2,3),(15,2,4),(15,3,1),(15,3,2),(15,3,3),(15,3,4),(16,1,1),(16,1,2),(16,1,3),(16,1,4),(16,2,1),(16,2,2),(16,2,3),(16,2,4),(16,3,1),(16,3,2),(16,3,3),(16,3,4),(17,1,1),(17,1,2),(17,1,3),(17,1,4),(17,2,1),(17,2,2),(17,2,3),(17,2,4),(17,3,1),(17,3,2),(17,3,3),(17,3,4),(18,1,1),(18,1,2),(18,1,3),(18,1,4),(18,2,1),(18,2,2),(18,2,3),(18,2,4),(18,3,1),(18,3,2),(18,3,3),(18,3,4),(19,1,1),(19,1,2),(19,1,3),(19,1,4),(19,2,1),(19,2,2),(19,2,3),(19,2,4),(19,3,1),(19,3,2),(19,3,3),(19,3,4),(20,1,1),(20,1,2),(20,1,3),(20,1,4),(20,2,1),(20,2,2),(20,2,3),(20,2,4),(20,3,1),(20,3,2),(20,3,3),(20,3,4),(21,1,1),(21,1,2),(21,1,3),(21,1,4),(21,2,1),(21,2,2),(21,2,3),(21,2,4),(21,3,1),(21,3,2),(21,3,3),(21,3,4),(22,1,1),(22,1,2),(22,1,3),(22,1,4),(22,2,1),(22,2,2),(22,2,3),(22,2,4),(22,3,1),(22,3,2),(22,3,3),(22,3,4),(23,1,1),(23,1,2),(23,1,3),(23,1,4),(23,2,1),(23,2,2),(23,2,3),(23,2,4),(23,3,1),(23,3,2),(23,3,3),(23,3,4),(24,1,1),(24,1,2),(24,1,3),(24,1,4),(24,2,1),(24,2,2),(24,2,3),(24,2,4),(24,3,1),(24,3,2),(24,3,3),(24,3,4),(25,1,1),(25,1,2),(25,1,3),(25,1,4),(25,2,1),(25,2,2),(25,2,3),(25,2,4),(25,3,1),(25,3,2),(25,3,3),(25,3,4),(26,1,1),(26,1,2),(26,1,3),(26,1,4),(26,2,1),(26,2,2),(26,2,3),(26,2,4),(26,3,1),(26,3,2),(26,3,3),(26,3,4),(27,1,1),(27,1,2),(27,1,3),(27,1,4),(27,2,1),(27,2,2),(27,2,3),(27,2,4),(27,3,1),(27,3,2),(27,3,3),(27,3,4),(28,1,1),(28,1,2),(28,1,3),(28,1,4),(28,2,1),(28,2,2),(28,2,3),(28,2,4),(28,3,1),(28,3,2),(28,3,3),(28,3,4),(29,1,1),(29,1,2),(29,1,3),(29,1,4),(29,2,1),(29,2,2),(29,2,3),(29,2,4),(29,3,1),(29,3,2),(29,3,3),(29,3,4),(30,1,1),(30,1,2),(30,1,3),(30,1,4),(30,2,1),(30,2,2),(30,2,3),(30,2,4),(30,3,1),(30,3,2),(30,3,3),(30,3,4),(31,1,1),(31,1,2),(31,1,3),(31,1,4),(31,2,1),(31,2,2),(31,2,3),(31,2,4),(31,3,1),(31,3,2),(31,3,3),(31,3,4),(32,1,1),(32,1,2),(32,1,3),(32,1,4),(32,2,1),(32,2,2),(32,2,3),(32,2,4),(32,3,1),(32,3,2),(32,3,3),(32,3,4),(33,1,1),(33,1,2),(33,1,3),(33,1,4),(33,2,1),(33,2,2),(33,2,3),(33,2,4),(33,3,1),(33,3,2),(33,3,3),(33,3,4),(34,1,1),(34,1,2),(34,1,3),(34,1,4),(34,2,1),(34,2,2),(34,2,3),(34,2,4),(34,3,1),(34,3,2),(34,3,3),(34,3,4),(35,1,1),(35,1,2),(35,1,3),(35,1,4),(35,2,1),(35,2,2),(35,2,3),(35,2,4),(35,3,1),(35,3,2),(35,3,3),(35,3,4),(36,1,1),(36,1,2),(36,1,3),(36,1,4),(36,2,1),(36,2,2),(36,2,3),(36,2,4),(36,3,1),(36,3,2),(36,3,3),(36,3,4),(37,1,1),(37,1,2),(37,1,3),(37,1,4),(37,2,1),(37,2,2),(37,2,3),(37,2,4),(37,3,1),(37,3,2),(37,3,3),(37,3,4),(38,1,1),(38,1,2),(38,1,3),(38,1,4),(38,2,1),(38,2,2),(38,2,3),(38,2,4),(38,3,1),(38,3,2),(38,3,3),(38,3,4),(39,1,1),(39,1,2),(39,1,3),(39,1,4),(39,2,1),(39,2,2),(39,2,3),(39,2,4),(39,3,1),(39,3,2),(39,3,3),(39,3,4),(40,1,1),(40,1,2),(40,1,3),(40,1,4),(40,2,1),(40,2,2),(40,2,3),(40,2,4),(40,3,1),(40,3,2),(40,3,3),(40,3,4),(41,1,1),(41,1,2),(41,1,3),(41,1,4),(41,2,1),(41,2,2),(41,2,3),(41,2,4),(41,3,1),(41,3,2),(41,3,3),(41,3,4),(42,1,1),(42,1,2),(42,1,3),(42,1,4),(42,2,1),(42,2,2),(42,2,3),(42,2,4),(42,3,1),(42,3,2),(42,3,3),(42,3,4),(43,1,1),(43,1,2),(43,1,3),(43,1,4),(43,2,1),(43,2,2),(43,2,3),(43,2,4),(43,3,1),(43,3,2),(43,3,3),(43,3,4),(44,1,1),(44,1,2),(44,1,3),(44,1,4),(44,2,1),(44,2,2),(44,2,3),(44,2,4),(44,3,1),(44,3,2),(44,3,3),(44,3,4),(45,1,1),(45,1,2),(45,1,3),(45,1,4),(45,2,1),(45,2,2),(45,2,3),(45,2,4),(45,3,1),(45,3,2),(45,3,3),(45,3,4),(46,1,1),(46,1,2),(46,1,3),(46,1,4),(46,2,1),(46,2,2),(46,2,3),(46,2,4),(46,3,1),(46,3,2),(46,3,3),(46,3,4),(47,1,1),(47,1,2),(47,1,3),(47,1,4),(47,2,1),(47,2,2),(47,2,3),(47,2,4),(47,3,1),(47,3,2),(47,3,3),(47,3,4),(48,1,1),(48,1,2),(48,1,3),(48,1,4),(48,2,1),(48,2,2),(48,2,3),(48,2,4),(48,3,1),(48,3,2),(48,3,3),(48,3,4),(49,1,1),(49,1,2),(49,1,3),(49,1,4),(49,2,1),(49,2,2),(49,2,3),(49,2,4),(49,3,1),(49,3,2),(49,3,3),(49,3,4),(50,1,1),(50,1,2),(50,1,3),(50,1,4),(50,2,1),(50,2,2),(50,2,3),(50,2,4),(50,3,1),(50,3,2),(50,3,3),(50,3,4),(51,1,1),(51,1,2),(51,1,3),(51,1,4),(51,2,1),(51,2,2),(51,2,3),(51,2,4),(51,3,1),(51,3,2),(51,3,3),(51,3,4),(52,1,1),(52,1,2),(52,1,3),(52,1,4),(52,2,1),(52,2,2),(52,2,3),(52,2,4),(52,3,1),(52,3,2),(52,3,3),(52,3,4),(53,1,1),(53,1,2),(53,1,3),(53,1,4),(53,2,1),(53,2,2),(53,2,3),(53,2,4),(53,3,1),(53,3,2),(53,3,3),(53,3,4),(54,1,1),(54,1,2),(54,1,3),(54,1,4),(54,2,1),(54,2,2),(54,2,3),(54,2,4),(54,3,1),(54,3,2),(54,3,3),(54,3,4),(55,1,1),(55,1,2),(55,1,3),(55,1,4),(55,2,1),(55,2,2),(55,2,3),(55,2,4),(55,3,1),(55,3,2),(55,3,3),(55,3,4),(56,1,1),(56,1,2),(56,1,3),(56,1,4),(56,2,1),(56,2,2),(56,2,3),(56,2,4),(56,3,1),(56,3,2),(56,3,3),(56,3,4),(57,1,1),(57,1,2),(57,1,3),(57,1,4),(57,2,1),(57,2,2),(57,2,3),(57,2,4),(57,3,1),(57,3,2),(57,3,3),(57,3,4),(58,1,1),(58,1,2),(58,1,3),(58,1,4),(58,2,1),(58,2,2),(58,2,3),(58,2,4),(58,3,1),(58,3,2),(58,3,3),(58,3,4),(59,1,1),(59,1,2),(59,1,3),(59,1,4),(59,2,1),(59,2,2),(59,2,3),(59,2,4),(59,3,1),(59,3,2),(59,3,3),(59,3,4),(60,1,1),(60,1,2),(60,1,3),(60,1,4),(60,2,1),(60,2,2),(60,2,3),(60,2,4),(60,3,1),(60,3,2),(60,3,3),(60,3,4),(61,1,1),(61,1,2),(61,1,3),(61,1,4),(61,2,1),(61,2,2),(61,2,3),(61,2,4),(61,3,1),(61,3,2),(61,3,3),(61,3,4),(62,1,1),(62,1,2),(62,1,3),(62,1,4),(62,2,1),(62,2,2),(62,2,3),(62,2,4),(62,3,1),(62,3,2),(62,3,3),(62,3,4),(64,1,1),(64,1,2),(64,1,3),(64,1,4),(64,2,1),(64,2,2),(64,2,3),(64,2,4),(64,3,1),(64,3,2),(64,3,3),(64,3,4),(65,1,1),(65,1,2),(65,1,3),(65,1,4),(65,2,1),(65,2,2),(65,2,3),(65,2,4),(65,3,1),(65,3,2),(65,3,3),(65,3,4),(66,1,1),(66,1,2),(66,1,3),(66,1,4),(66,2,1),(66,2,2),(66,2,3),(66,2,4),(66,3,1),(66,3,2),(66,3,3),(66,3,4),(67,1,1),(67,1,2),(67,1,3),(67,1,4),(67,2,1),(67,2,2),(67,2,3),(67,2,4),(67,3,1),(67,3,2),(67,3,3),(67,3,4),(68,1,1),(68,1,2),(68,1,3),(68,1,4),(68,2,1),(68,2,2),(68,2,3),(68,2,4),(68,3,1),(68,3,2),(68,3,3),(68,3,4),(69,1,1),(69,1,2),(69,1,3),(69,1,4),(69,2,1),(69,2,2),(69,2,3),(69,2,4),(69,3,1),(69,3,2),(69,3,3),(69,3,4),(70,1,1),(70,1,2),(70,1,3),(70,1,4),(70,2,1),(70,2,2),(70,2,3),(70,2,4),(70,3,1),(70,3,2),(70,3,3),(70,3,4),(71,1,1),(71,1,2),(71,1,3),(71,1,4),(71,2,1),(71,2,2),(71,2,3),(71,2,4),(71,3,1),(71,3,2),(71,3,3),(71,3,4),(72,1,1),(72,1,2),(72,1,3),(72,1,4),(72,2,1),(72,2,2),(72,2,3),(72,2,4),(72,3,1),(72,3,2),(72,3,3),(72,3,4),(73,1,1),(73,1,2),(73,1,3),(73,1,4),(73,2,1),(73,2,2),(73,2,3),(73,2,4),(73,3,1),(73,3,2),(73,3,3),(73,3,4),(74,1,1),(74,1,2),(74,1,3),(74,1,4),(74,2,1),(74,2,2),(74,2,3),(74,2,4),(74,3,1),(74,3,2),(74,3,3),(74,3,4),(75,1,1),(75,1,2),(75,1,3),(75,1,4),(75,2,1),(75,2,2),(75,2,3),(75,2,4),(75,3,1),(75,3,2),(75,3,3),(75,3,4),(76,1,1),(76,1,2),(76,1,3),(76,1,4),(76,2,1),(76,2,2),(76,2,3),(76,2,4),(76,3,1),(76,3,2),(76,3,3),(76,3,4),(77,1,1),(77,1,2),(77,1,3),(77,1,4),(77,2,1),(77,2,2),(77,2,3),(77,2,4),(77,3,1),(77,3,2),(77,3,3),(77,3,4),(78,1,1),(78,1,2),(78,1,3),(78,1,4),(78,2,1),(78,2,2),(78,2,3),(78,2,4),(78,3,1),(78,3,2),(78,3,3),(78,3,4),(79,1,1),(79,1,2),(79,1,3),(79,1,4),(79,2,1),(79,2,2),(79,2,3),(79,2,4),(79,3,1),(79,3,2),(79,3,3),(79,3,4),(80,1,1),(80,1,2),(80,1,3),(80,1,4),(80,2,1),(80,2,2),(80,2,3),(80,2,4),(80,3,1),(80,3,2),(80,3,3),(80,3,4),(81,1,1),(81,1,2),(81,1,3),(81,1,4),(81,2,1),(81,2,2),(81,2,3),(81,2,4),(81,3,1),(81,3,2),(81,3,3),(81,3,4),(82,1,1),(82,1,2),(82,1,3),(82,1,4),(82,2,1),(82,2,2),(82,2,3),(82,2,4),(82,3,1),(82,3,2),(82,3,3),(82,3,4),(83,1,1),(83,1,2),(83,1,3),(83,1,4),(83,2,1),(83,2,2),(83,2,3),(83,2,4),(83,3,1),(83,3,2),(83,3,3),(83,3,4),(84,1,1),(84,1,2),(84,1,3),(84,1,4),(84,2,1),(84,2,2),(84,2,3),(84,2,4),(84,3,1),(84,3,2),(84,3,3),(84,3,4),(85,1,1),(85,1,2),(85,1,3),(85,1,4),(85,2,1),(85,2,2),(85,2,3),(85,2,4),(85,3,1),(85,3,2),(85,3,3),(85,3,4),(86,1,1),(86,1,2),(86,1,3),(86,1,4),(86,2,1),(86,2,2),(86,2,3),(86,2,4),(86,3,1),(86,3,2),(86,3,3),(86,3,4),(87,1,1),(87,1,2),(87,1,3),(87,1,4),(87,2,1),(87,2,2),(87,2,3),(87,2,4),(87,3,1),(87,3,2),(87,3,3),(87,3,4),(88,1,1),(88,1,2),(88,1,3),(88,1,4),(88,2,1),(88,2,2),(88,2,3),(88,2,4),(88,3,1),(88,3,2),(88,3,3),(88,3,4),(89,1,1),(89,1,2),(89,1,3),(89,1,4),(89,2,1),(89,2,2),(89,2,3),(89,2,4),(89,3,1),(89,3,2),(89,3,3),(89,3,4),(90,1,1),(90,1,2),(90,1,3),(90,1,4),(90,2,1),(90,2,2),(90,2,3),(90,2,4),(90,3,1),(90,3,2),(90,3,3),(90,3,4),(91,1,1),(91,1,2),(91,1,3),(91,1,4),(91,2,1),(91,2,2),(91,2,3),(91,2,4),(91,3,1),(91,3,2),(91,3,3),(91,3,4),(92,1,1),(92,1,2),(92,1,3),(92,1,4),(92,2,1),(92,2,2),(92,2,3),(92,2,4),(92,3,1),(92,3,2),(92,3,3),(92,3,4),(93,1,1),(93,1,2),(93,1,3),(93,1,4),(93,2,1),(93,2,2),(93,2,3),(93,2,4),(93,3,1),(93,3,2),(93,3,3),(93,3,4),(94,1,1),(94,1,2),(94,1,3),(94,1,4),(94,2,1),(94,2,2),(94,2,3),(94,2,4),(94,3,1),(94,3,2),(94,3,3),(94,3,4),(95,1,1),(95,1,2),(95,1,3),(95,1,4),(95,2,1),(95,2,2),(95,2,3),(95,2,4),(95,3,1),(95,3,2),(95,3,3),(95,3,4),(96,1,1),(96,1,2),(96,1,3),(96,1,4),(96,2,1),(96,2,2),(96,2,3),(96,2,4),(96,3,1),(96,3,2),(96,3,3),(96,3,4),(97,1,1),(97,1,2),(97,1,3),(97,1,4),(97,2,1),(97,2,2),(97,2,3),(97,2,4),(97,3,1),(97,3,2),(97,3,3),(97,3,4),(98,1,1),(98,1,2),(98,1,3),(98,1,4),(98,2,1),(98,2,2),(98,2,3),(98,2,4),(98,3,1),(98,3,2),(98,3,3),(98,3,4),(99,1,1),(99,1,2),(99,1,3),(99,1,4),(99,2,1),(99,2,2),(99,2,3),(99,2,4),(99,3,1),(99,3,2),(99,3,3),(99,3,4),(100,1,1),(100,1,2),(100,1,3),(100,1,4),(100,2,1),(100,2,2),(100,2,3),(100,2,4),(100,3,1),(100,3,2),(100,3,3),(100,3,4),(101,1,1),(101,1,2),(101,1,3),(101,1,4),(101,2,1),(101,2,2),(101,2,3),(101,2,4),(101,3,1),(101,3,2),(101,3,3),(101,3,4),(102,1,1),(102,1,2),(102,1,3),(102,1,4),(102,2,1),(102,2,2),(102,2,3),(102,2,4),(102,3,1),(102,3,2),(102,3,3),(102,3,4),(103,1,1),(103,1,2),(103,1,3),(103,1,4),(103,2,1),(103,2,2),(103,2,3),(103,2,4),(103,3,1),(103,3,2),(103,3,3),(103,3,4),(104,1,1),(104,1,2),(104,1,3),(104,1,4),(104,2,1),(104,2,2),(104,2,3),(104,2,4),(104,3,1),(104,3,2),(104,3,3),(104,3,4),(105,1,1),(105,1,2),(105,1,3),(105,1,4),(105,2,1),(105,2,2),(105,2,3),(105,2,4),(105,3,1),(105,3,2),(105,3,3),(105,3,4),(106,1,1),(106,1,2),(106,1,3),(106,1,4),(106,2,1),(106,2,2),(106,2,3),(106,2,4),(106,3,1),(106,3,2),(106,3,3),(106,3,4),(107,1,1),(107,1,2),(107,1,3),(107,1,4),(107,2,1),(107,2,2),(107,2,3),(107,2,4),(107,3,1),(107,3,2),(107,3,3),(107,3,4),(108,1,1),(108,1,2),(108,1,3),(108,1,4),(108,2,1),(108,2,2),(108,2,3),(108,2,4),(108,3,1),(108,3,2),(108,3,3),(108,3,4),(109,1,1),(109,1,2),(109,1,3),(109,1,4),(109,2,1),(109,2,2),(109,2,3),(109,2,4),(109,3,1),(109,3,2),(109,3,3),(109,3,4),(110,1,1),(110,1,2),(110,1,3),(110,1,4),(110,2,1),(110,2,2),(110,2,3),(110,2,4),(110,3,1),(110,3,2),(110,3,3),(110,3,4),(111,1,1),(111,1,2),(111,1,3),(111,1,4),(111,2,1),(111,2,2),(111,2,3),(111,2,4),(111,3,1),(111,3,2),(111,3,3),(111,3,4),(112,1,1),(112,1,2),(112,1,3),(112,1,4),(112,2,1),(112,2,2),(112,2,3),(112,2,4),(112,3,1),(112,3,2),(112,3,3),(112,3,4),(113,1,1),(113,1,2),(113,1,3),(113,1,4),(113,2,1),(113,2,2),(113,2,3),(113,2,4),(113,3,1),(113,3,2),(113,3,3),(113,3,4),(114,1,1),(114,1,2),(114,1,3),(114,1,4),(114,2,1),(114,2,2),(114,2,3),(114,2,4),(114,3,1),(114,3,2),(114,3,3),(114,3,4),(115,1,1),(115,1,2),(115,1,3),(115,1,4),(115,2,1),(115,2,2),(115,2,3),(115,2,4),(115,3,1),(115,3,2),(115,3,3),(115,3,4),(116,1,1),(116,1,2),(116,1,3),(116,1,4),(116,2,1),(116,2,2),(116,2,3),(116,2,4),(116,3,1),(116,3,2),(116,3,3),(116,3,4),(117,1,1),(117,1,2),(117,1,3),(117,1,4),(117,2,1),(117,2,2),(117,2,3),(117,2,4),(117,3,1),(117,3,2),(117,3,3),(117,3,4),(118,1,1),(118,1,2),(118,1,3),(118,1,4),(118,2,1),(118,2,2),(118,2,3),(118,2,4),(118,3,1),(118,3,2),(118,3,3),(118,3,4),(119,1,1),(119,1,2),(119,1,3),(119,1,4),(119,2,1),(119,2,2),(119,2,3),(119,2,4),(119,3,1),(119,3,2),(119,3,3),(119,3,4),(120,1,1),(120,1,2),(120,1,3),(120,1,4),(120,2,1),(120,2,2),(120,2,3),(120,2,4),(120,3,1),(120,3,2),(120,3,3),(120,3,4),(121,1,1),(121,1,2),(121,1,3),(121,1,4),(121,2,1),(121,2,2),(121,2,3),(121,2,4),(121,3,1),(121,3,2),(121,3,3),(121,3,4),(122,1,1),(122,1,2),(122,1,3),(122,1,4),(122,2,1),(122,2,2),(122,2,3),(122,2,4),(122,3,1),(122,3,2),(122,3,3),(122,3,4),(123,1,1),(123,1,2),(123,1,3),(123,1,4),(123,2,1),(123,2,2),(123,2,3),(123,2,4),(123,3,1),(123,3,2),(123,3,3),(123,3,4),(124,1,1),(124,1,2),(124,1,3),(124,1,4),(124,2,1),(124,2,2),(124,2,3),(124,2,4),(124,3,1),(124,3,2),(124,3,3),(124,3,4),(125,1,1),(125,1,2),(125,1,3),(125,1,4),(125,2,1),(125,2,2),(125,2,3),(125,2,4),(125,3,1),(125,3,2),(125,3,3),(125,3,4),(126,1,1),(126,1,2),(126,1,3),(126,1,4),(126,2,1),(126,2,2),(126,2,3),(126,2,4),(126,3,1),(126,3,2),(126,3,3),(126,3,4),(127,1,1),(127,1,2),(127,1,3),(127,1,4),(127,2,1),(127,2,2),(127,2,3),(127,2,4),(127,3,1),(127,3,2),(127,3,3),(127,3,4),(128,1,1),(128,1,2),(128,1,3),(128,1,4),(128,2,1),(128,2,2),(128,2,3),(128,2,4),(128,3,1),(128,3,2),(128,3,3),(128,3,4),(129,1,1),(129,1,2),(129,1,3),(129,1,4),(129,2,1),(129,2,2),(129,2,3),(129,2,4),(129,3,1),(129,3,2),(129,3,3),(129,3,4),(130,1,1),(130,1,2),(130,1,3),(130,1,4),(130,2,1),(130,2,2),(130,2,3),(130,2,4),(130,3,1),(130,3,2),(130,3,3),(130,3,4),(131,1,1),(131,1,2),(131,1,3),(131,1,4),(131,2,1),(131,2,2),(131,2,3),(131,2,4),(131,3,1),(131,3,2),(131,3,3),(131,3,4),(132,1,1),(132,1,2),(132,1,3),(132,1,4),(132,2,1),(132,2,2),(132,2,3),(132,2,4),(132,3,1),(132,3,2),(132,3,3),(132,3,4),(133,1,1),(133,1,2),(133,1,3),(133,1,4),(133,2,1),(133,2,2),(133,2,3),(133,2,4),(133,3,1),(133,3,2),(133,3,3),(133,3,4),(134,1,1),(134,1,2),(134,1,3),(134,1,4),(134,2,1),(134,2,2),(134,2,3),(134,2,4),(134,3,1),(134,3,2),(134,3,3),(134,3,4),(135,1,1),(135,1,2),(135,1,3),(135,1,4),(135,2,1),(135,2,2),(135,2,3),(135,2,4),(135,3,1),(135,3,2),(135,3,3),(135,3,4),(136,1,1),(136,1,2),(136,1,3),(136,1,4),(136,2,1),(136,2,2),(136,2,3),(136,2,4),(136,3,1),(136,3,2),(136,3,3),(136,3,4),(137,1,1),(137,1,2),(137,1,3),(137,1,4),(137,2,1),(137,2,2),(137,2,3),(137,2,4),(137,3,1),(137,3,2),(137,3,3),(137,3,4),(138,1,1),(138,1,2),(138,1,3),(138,1,4),(138,2,1),(138,2,2),(138,2,3),(138,2,4),(138,3,1),(138,3,2),(138,3,3),(138,3,4),(139,1,1),(139,1,2),(139,1,3),(139,1,4),(139,2,1),(139,2,2),(139,2,3),(139,2,4),(139,3,1),(139,3,2),(139,3,3),(139,3,4),(140,1,1),(140,1,2),(140,1,3),(140,1,4),(140,2,1),(140,2,2),(140,2,3),(140,2,4),(140,3,1),(140,3,2),(140,3,3),(140,3,4),(141,1,1),(141,1,2),(141,1,3),(141,1,4),(141,2,1),(141,2,2),(141,2,3),(141,2,4),(141,3,1),(141,3,2),(141,3,3),(141,3,4),(142,1,1),(142,1,2),(142,1,3),(142,1,4),(142,2,1),(142,2,2),(142,2,3),(142,2,4),(142,3,1),(142,3,2),(142,3,3),(142,3,4),(143,1,1),(143,1,2),(143,1,3),(143,1,4),(143,2,1),(143,2,2),(143,2,3),(143,2,4),(143,3,1),(143,3,2),(143,3,3),(143,3,4),(144,1,1),(144,1,2),(144,1,3),(144,1,4),(144,2,1),(144,2,2),(144,2,3),(144,2,4),(144,3,1),(144,3,2),(144,3,3),(144,3,4),(145,1,1),(145,1,2),(145,1,3),(145,1,4),(145,2,1),(145,2,2),(145,2,3),(145,2,4),(145,3,1),(145,3,2),(145,3,3),(145,3,4),(146,1,1),(146,1,2),(146,1,3),(146,1,4),(146,2,1),(146,2,2),(146,2,3),(146,2,4),(146,3,1),(146,3,2),(146,3,3),(146,3,4),(147,1,1),(147,1,2),(147,1,3),(147,1,4),(147,2,1),(147,2,2),(147,2,3),(147,2,4),(147,3,1),(147,3,2),(147,3,3),(147,3,4),(148,1,1),(148,1,2),(148,1,3),(148,1,4),(148,2,1),(148,2,2),(148,2,3),(148,2,4),(148,3,1),(148,3,2),(148,3,3),(148,3,4),(149,1,1),(149,1,2),(149,1,3),(149,1,4),(149,2,1),(149,2,2),(149,2,3),(149,2,4),(149,3,1),(149,3,2),(149,3,3),(149,3,4),(150,1,1),(150,1,2),(150,1,3),(150,1,4),(150,2,1),(150,2,2),(150,2,3),(150,2,4),(150,3,1),(150,3,2),(150,3,3),(150,3,4),(151,1,1),(151,1,2),(151,1,3),(151,1,4),(151,2,1),(151,2,2),(151,2,3),(151,2,4),(151,3,1),(151,3,2),(151,3,3),(151,3,4),(152,1,1),(152,1,2),(152,1,3),(152,1,4),(152,2,1),(152,2,2),(152,2,3),(152,2,4),(152,3,1),(152,3,2),(152,3,3),(152,3,4),(153,1,1),(153,1,2),(153,1,3),(153,1,4),(153,2,1),(153,2,2),(153,2,3),(153,2,4),(153,3,1),(153,3,2),(153,3,3),(153,3,4),(154,1,1),(154,1,2),(154,1,3),(154,1,4),(154,2,1),(154,2,2),(154,2,3),(154,2,4),(154,3,1),(154,3,2),(154,3,3),(154,3,4),(155,1,1),(155,1,2),(155,1,3),(155,1,4),(155,2,1),(155,2,2),(155,2,3),(155,2,4),(155,3,1),(155,3,2),(155,3,3),(155,3,4),(156,1,1),(156,1,2),(156,1,3),(156,1,4),(156,2,1),(156,2,2),(156,2,3),(156,2,4),(156,3,1),(156,3,2),(156,3,3),(156,3,4),(157,1,1),(157,1,2),(157,1,3),(157,1,4),(157,2,1),(157,2,2),(157,2,3),(157,2,4),(157,3,1),(157,3,2),(157,3,3),(157,3,4),(158,1,1),(158,1,2),(158,1,3),(158,1,4),(158,2,1),(158,2,2),(158,2,3),(158,2,4),(158,3,1),(158,3,2),(158,3,3),(158,3,4),(159,1,1),(159,1,2),(159,1,3),(159,1,4),(159,2,1),(159,2,2),(159,2,3),(159,2,4),(159,3,1),(159,3,2),(159,3,3),(159,3,4),(160,1,1),(160,1,2),(160,1,3),(160,1,4),(160,2,1),(160,2,2),(160,2,3),(160,2,4),(160,3,1),(160,3,2),(160,3,3),(160,3,4),(161,1,1),(161,1,2),(161,1,3),(161,1,4),(161,2,1),(161,2,2),(161,2,3),(161,2,4),(161,3,1),(161,3,2),(161,3,3),(161,3,4),(162,1,1),(162,1,2),(162,1,3),(162,1,4),(162,2,1),(162,2,2),(162,2,3),(162,2,4),(162,3,1),(162,3,2),(162,3,3),(162,3,4),(163,1,1),(163,1,2),(163,1,3),(163,1,4),(163,2,1),(163,2,2),(163,2,3),(163,2,4),(163,3,1),(163,3,2),(163,3,3),(163,3,4),(164,1,1),(164,1,2),(164,1,3),(164,1,4),(164,2,1),(164,2,2),(164,2,3),(164,2,4),(164,3,1),(164,3,2),(164,3,3),(164,3,4),(165,1,1),(165,1,2),(165,1,3),(165,1,4),(165,2,1),(165,2,2),(165,2,3),(165,2,4),(165,3,1),(165,3,2),(165,3,3),(165,3,4),(166,1,1),(166,1,2),(166,1,3),(166,1,4),(166,2,1),(166,2,2),(166,2,3),(166,2,4),(166,3,1),(166,3,2),(166,3,3),(166,3,4),(167,1,1),(167,1,2),(167,1,3),(167,1,4),(167,2,1),(167,2,2),(167,2,3),(167,2,4),(167,3,1),(167,3,2),(167,3,3),(167,3,4),(168,1,1),(168,1,2),(168,1,3),(168,1,4),(168,2,1),(168,2,2),(168,2,3),(168,2,4),(168,3,1),(168,3,2),(168,3,3),(168,3,4),(169,1,1),(169,1,2),(169,1,3),(169,1,4),(169,2,1),(169,2,2),(169,2,3),(169,2,4),(169,3,1),(169,3,2),(169,3,3),(169,3,4),(170,1,1),(170,1,2),(170,1,3),(170,1,4),(170,2,1),(170,2,2),(170,2,3),(170,2,4),(170,3,1),(170,3,2),(170,3,3),(170,3,4),(171,1,1),(171,1,2),(171,1,3),(171,1,4),(171,2,1),(171,2,2),(171,2,3),(171,2,4),(171,3,1),(171,3,2),(171,3,3),(171,3,4),(172,1,1),(172,1,2),(172,1,3),(172,1,4),(172,2,1),(172,2,2),(172,2,3),(172,2,4),(172,3,1),(172,3,2),(172,3,3),(172,3,4),(173,1,1),(173,1,2),(173,1,3),(173,1,4),(173,2,1),(173,2,2),(173,2,3),(173,2,4),(173,3,1),(173,3,2),(173,3,3),(173,3,4),(174,1,1),(174,1,2),(174,1,3),(174,1,4),(174,2,1),(174,2,2),(174,2,3),(174,2,4),(174,3,1),(174,3,2),(174,3,3),(174,3,4),(175,1,1),(175,1,2),(175,1,3),(175,1,4),(175,2,1),(175,2,2),(175,2,3),(175,2,4),(175,3,1),(175,3,2),(175,3,3),(175,3,4),(176,1,1),(176,1,2),(176,1,3),(176,1,4),(176,2,1),(176,2,2),(176,2,3),(176,2,4),(176,3,1),(176,3,2),(176,3,3),(176,3,4),(177,1,1),(177,1,2),(177,1,3),(177,1,4),(177,2,1),(177,2,2),(177,2,3),(177,2,4),(177,3,1),(177,3,2),(177,3,3),(177,3,4),(178,1,1),(178,1,2),(178,1,3),(178,1,4),(178,2,1),(178,2,2),(178,2,3),(178,2,4),(178,3,1),(178,3,2),(178,3,3),(178,3,4),(179,1,1),(179,1,2),(179,1,3),(179,1,4),(179,2,1),(179,2,2),(179,2,3),(179,2,4),(179,3,1),(179,3,2),(179,3,3),(179,3,4),(180,1,1),(180,1,2),(180,1,3),(180,1,4),(180,2,1),(180,2,2),(180,2,3),(180,2,4),(180,3,1),(180,3,2),(180,3,3),(180,3,4),(181,1,1),(181,1,2),(181,1,3),(181,1,4),(181,2,1),(181,2,2),(181,2,3),(181,2,4),(181,3,1),(181,3,2),(181,3,3),(181,3,4),(182,1,1),(182,1,2),(182,1,3),(182,1,4),(182,2,1),(182,2,2),(182,2,3),(182,2,4),(182,3,1),(182,3,2),(182,3,3),(182,3,4),(183,1,1),(183,1,2),(183,1,3),(183,1,4),(183,2,1),(183,2,2),(183,2,3),(183,2,4),(183,3,1),(183,3,2),(183,3,3),(183,3,4),(184,1,1),(184,1,2),(184,1,3),(184,1,4),(184,2,1),(184,2,2),(184,2,3),(184,2,4),(184,3,1),(184,3,2),(184,3,3),(184,3,4),(185,1,1),(185,1,2),(185,1,3),(185,1,4),(185,2,1),(185,2,2),(185,2,3),(185,2,4),(185,3,1),(185,3,2),(185,3,3),(185,3,4),(186,1,1),(186,1,2),(186,1,3),(186,1,4),(186,2,1),(186,2,2),(186,2,3),(186,2,4),(186,3,1),(186,3,2),(186,3,3),(186,3,4),(187,1,1),(187,1,2),(187,1,3),(187,1,4),(187,2,1),(187,2,2),(187,2,3),(187,2,4),(187,3,1),(187,3,2),(187,3,3),(187,3,4),(188,1,1),(188,1,2),(188,1,3),(188,1,4),(188,2,1),(188,2,2),(188,2,3),(188,2,4),(188,3,1),(188,3,2),(188,3,3),(188,3,4),(189,1,1),(189,1,2),(189,1,3),(189,1,4),(189,2,1),(189,2,2),(189,2,3),(189,2,4),(189,3,1),(189,3,2),(189,3,3),(189,3,4);
/*!40000 ALTER TABLE `signals` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-06-05  7:45:31
