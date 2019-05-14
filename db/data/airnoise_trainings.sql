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
-- Dumping data for table `trainings`
--

LOCK TABLES `trainings` WRITE;
/*!40000 ALTER TABLE `trainings` DISABLE KEYS */;
INSERT INTO `trainings` (`id_training`, `description`) VALUES (1,'Spatial Uncorrected (S1)'),(2,'Spatial Uncorrected (S2)'),(3,'Spatial Uncorrected (S3)'),(4,'Spatial Uncorrected (S4)'),(5,'Spatial Uncorrected V2 (S1)'),(6,'Spatial Uncorrected V2 (S2)'),(7,'Spatial Uncorrected V2 (S3)'),(8,'Spatial Uncorrected V2 (S4)'),(9,'Non-spatial Uncorrected V2 (S1)'),(10,'Non-spatial Uncorrected V2 (S2)'),(11,'Non-spatial Uncorrected V2 (S3)'),(12,'Non-spatial Uncorrected V2 (S4)'),(13,'Spatial 512  FFT (S1)'),(14,'Spatial 512  FFT (S2)'),(15,'Spatial 512  FFT (S3)'),(16,'Spatial 512  FFT (S4)'),(17,'Non-spatial 512 FFT (S1)'),(18,'Non-spatial 512 FFT (S2)'),(19,'Non-spatial 512 FFT (S3)'),(20,'Non-spatial 512 FFT (S4)');
/*!40000 ALTER TABLE `trainings` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-05-14  9:44:10
