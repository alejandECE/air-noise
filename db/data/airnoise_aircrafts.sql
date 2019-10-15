-- MySQL dump 10.13  Distrib 8.0.17, for Win64 (x86_64)
--
-- Host: localhost    Database: airnoise
-- ------------------------------------------------------
-- Server version	8.0.17

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Dumping data for table `aircrafts`
--

LOCK TABLES `aircrafts` WRITE;
/*!40000 ALTER TABLE `aircrafts` DISABLE KEYS */;
INSERT INTO `aircrafts` VALUES (0,'A340-6* (Trent 556A2-61)','In the ANP appears in the acoustic class TRENT5.'),(1,'A320-2xx (CFM56-5)','In the ANP the A320-211 is in the acoustic class CFM565. Airbus builds the A320-2 * with 2 types of engines.'),(2,'A320-2xx (V25xx)','In the ANP the A320-232 is in the acoustic class V2527A. Airbus builds the A320-2 * with 2 types of engines.'),(3,'A319-1xx (CFM56-5)','In the ANP only the A319-131 with V2522-A5 engines appears but there are 2 variants.'),(4,'A319-1xx (V25xx)','In the ANP appears the A319-131 with V2522-A5 engines. Airbus builds the A319 with CFM56-5B and V2500-A5 engines.'),(5,'ATR42-300 (PW120)','In the ANP it does not appear but there is a class PW120 to which the DASH 8-100 and DASH 8-300 belong.'),(6,'ATR42-500 (PW127E)','It does not appear in the ANP.'),(7,'ATR72-600 (PW127M)','It does not appear in the ANP.'),(8,'B737-3xx (CFM56-3)','In the ANP both 737-300 and 737-3B2 appear in the acoustic class CFM563.'),(9,'B737-8xx (CF56-7B22+)','In the ANP both 737-700 and 737-800 are mapped to the same acoustic class CF567B.'),(10,'B737-7xx (CF56-7B22-)','In the ANP both 737-700 and 737-800 are mapped to the same acoustic class CF567B.'),(11,'ERJ145 (AE3007)','In the ANP both EMB145 and EMB14L are mapped to the acoustic class AE3007.'),(12,'ERJ170/175 (CF34-8E)','It does not appear in the ANP.'),(13,'ERJ190 (CF34-10E)','It does not appear in the ANP.'),(14,'MD-8x (2JT8D2)','In the ANP aircrafts MD81, MD82 and MD83 are grouped in the same acoustic class 2JT8D2.'),(15,'TU204 (PS-90A)','It does not appear in the ANP.'),(16,'SU95 (SaM 146)','It does not appear in the ANP. Sukoi.'),(17,'787-8 Dreamliner (GEnx-1B)','It does not appear in the ANP.'),(18,'B767-3* (PW4060)','In the ANP appears in the acoustic class 2CF680.'),(19,'B747-8* (GEnx-2B67)','It does not appear in the ANP.'),(20,'B747-4*(CF6-80C2B5F)','It does not appear in the ANP.'),(21,'CRJ200 (CF34-3B1)','It does not appear in the ANP.');
/*!40000 ALTER TABLE `aircrafts` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-08-11 20:29:54
