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
-- Dumping data for table `calibrations`
--

LOCK TABLES `calibrations` WRITE;
/*!40000 ALTER TABLE `calibrations` DISABLE KEYS */;
INSERT INTO `calibrations` (`id_calibration`, `date`, `value`, `url`, `description`) VALUES (1,'2013-03-23',39.4855,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d1 a0.lvm','Channel 0 Measurements taken 2013-03-23'),(2,NULL,38.3536,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d1 a1.lvm','Channel 1 Measurements taken 2013-03-23'),(3,NULL,32.4266,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d1 a2.lvm','Channel 2 Measurements taken 2013-03-23'),(4,NULL,38.6542,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d1 a3.lvm','Channel 3 Measurements taken 2013-03-23'),(5,NULL,62.4705,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d2 a0.lvm','Channel 4 Measurements taken 2013-03-23'),(6,NULL,25.8494,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d2 a1.lvm','Channel 5 Measurements taken 2013-03-23'),(7,NULL,56.3808,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d2 a2.lvm','Channel 6 Measurements taken 2013-03-23'),(8,NULL,36.3651,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d2 a3.lvm','Channel 7 Measurements taken 2013-03-23'),(9,NULL,44.1533,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d3 a0.lvm','Channel 8 Measurements taken 2013-03-23'),(10,NULL,41.5377,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d3 a1.lvm','Channel 9 Measurements taken 2013-03-23'),(11,NULL,51.2477,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d3 a2.lvm','Channel 10 Measurements taken 2013-03-23'),(12,NULL,42.9976,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-03-23\\calibration\\d3 a3.lvm','Channel 11 Measurements taken 2013-03-23'),(13,NULL,39.2418,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\1 v 1.lvm','Channel 0 Measurements taken 2013-09-13'),(14,NULL,39.6058,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\2 v 2.lvm','Channel 1 Measurements taken 2013-09-13'),(15,NULL,33.0075,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\3 v 3.lvm','Channel 2 Measurements taken 2013-09-13'),(16,NULL,57.4895,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\4 v 4.lvm','Channel 3 Measurements taken 2013-09-13'),(17,NULL,46.6512,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\5 f 1.lvm','Channel 4 Measurements taken 2013-09-13'),(18,NULL,42.9237,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\6 f 2.lvm','Channel 5 Measurements taken 2013-09-13'),(19,NULL,50.9053,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\7 f 3.lvm','Channel 6 Measurements taken 2013-09-13'),(20,NULL,43.0711,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\8 f 4.lvm','Channel 7 Measurements taken 2013-09-13'),(21,NULL,62.7494,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\9 h 1.lvm','Channel 8 Measurements taken 2013-09-13'),(22,NULL,26.3591,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\10 h 2.lvm','Channel 9 Measurements taken 2013-09-13'),(23,NULL,53.4397,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\11 h 3.lvm','Channel 10 Measurements taken 2013-09-13'),(24,NULL,37.6973,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2013-09-13\\calibration (GOOD)\\12 h 4.lvm','Channel 11 Measurements taken 2013-09-13'),(25,NULL,38.65,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 0 Measurements taken 2014-02-21'),(26,NULL,38.67,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 1 Measurements taken 2014-02-21'),(27,NULL,32.3,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 2 Measurements taken 2014-02-21'),(28,NULL,42.2,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 3 Measurements taken 2014-02-21'),(29,NULL,61.8,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 4 Measurements taken 2014-02-21'),(30,NULL,25.69,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 5 Measurements taken 2014-02-21'),(31,NULL,51.8,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 6 Measurements taken 2014-02-21'),(32,NULL,34.63,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 7 Measurements taken 2014-02-21'),(33,NULL,42.57,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 8 Measurements taken 2014-02-21'),(34,NULL,39.7,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 9 Measurements taken 2014-02-21'),(35,NULL,46,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 10 Measurements taken 2014-02-21'),(36,NULL,41.46,'D:\\Luis Alejandro\\Doctorate (Ph.D)\\Data\\Measurements (Ph.D)\\2014-02-21\\calibration\\CALIBRACION USADA.jpg','Channel 11 Measurements taken 2014-02-21');
/*!40000 ALTER TABLE `calibrations` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-05-14  9:44:17
