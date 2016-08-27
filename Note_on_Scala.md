#Note on Scala
Sometimes it is hard to classify whether a feature belongs to FP or specific to Scala.  Therefore I made a section here.

##Using Scala Stream
```Stream``` implements lazy lists where elements are evaluated only when they are needed.  This uses the memory more efficiently.  But using ```Stream``` can also result in out of memory problem.  Once an item is evaluated, memory will be allocated to hold it.  To discard an element after evaluation, don't use a reference to hold a ```Stream```.  E.g. 
```scala
scala> def stream: Stream[String] = file.getLines.toStream
scala> stream.foldLeft(zero)(_ + _.toLong)
res1: scala.math.BigDecimal = 89148000000000000000

scala> val s = stream
scala> s.foldLeft(zero)(_ + _.toLong)
java.lang.OutOfMemoryError: Java heap space
```
A reference ```S``` holds the stream.  Therefore it holds all the evaluated items.

An alternative solution is using ```iterator``` on the ```Stream``` to evaluate the items.

##Using Future in an Actor application
In Scala, ```Future``` is implemented as a Monad.  Different Monads have different behaviours to serve different purposes.

There is something interesting about ```Future``` as a Monad.  It is a placeholder for a function result that will be available at some point in the future.  In an Actor application, it is common that a ```var``` is used to maintain the Actor state.  In some scenarios, a ```Future``` is used together with this Actor where a function is performed inside the Future.  Under this condition, it should be careful in using this ```var```.  For example,
```Scala
class WriteJournalEntryToPostgres extends Actor {
  var seqNrMap = Map.empty[String, Long]
  
  val conn: PostgreSQLConnection = ???
  
  var future = conn.connect.flatMap {
    _ => Future(List.empty[QueryResult])
  }
  
  val table: String = context.system.settings.config.getString("akka-persistence-sql-async.journal-table-name")
 
  var noOfEntries = 0l
  
  var insertValuesForBatch = List.empty[String]
  
  var paramsForBatch = List.empty[Any]
  
  val batchSize = 900
  
  val insertStmt = s"insert into $table (persistence_id, sequence_nr, message, marker, created_at) values "
 
  def arEventMatch(seqNr: Long): PartialFunction[EntityEvent, Try[BasePersistWrapper[_]]] = ???
 
  def updateDb(persistenceId: String, seqNr: Long, message: Array[Byte]): Unit = {
    insertValuesForBatch = "(?, ?, ?, 'A', CURRENT_TIMESTAMP)" :: insertValuesForBatch
    paramsForBatch = paramsForBatch ::: List(persistenceId, seqNr, message)
    if (insertValuesForBatch.size == batchSize) 
      executePstmt()
  }
  
  def executePstmt(): Unit = {
    val pStmt = List(insertStmt, insertValuesForBatch.mkString(",\n")).mkString
    val params = paramsForBatch
    future = future flatMap {
      qrs => conn.sendPreparedStatement(pStmt, params) map (qrs ::: List(_))
    }
    insertValuesForBatch = Nil
    paramsForBatch = Nil
  }
 
  def processPersistentRepr(id: String)(origPr: PersistentRepr)(objType: String)(`====`: String)(obj: AnyRef): Unit = {
    val seqNr = seqNrMap.getOrElse(origPr.persistenceId, 0l)
    seqNrMap = seqNrMap + (origPr.persistenceId -> (seqNr + 1))
    val persistRepr = PersistentRepr(
      payload = obj,
      sequenceNr = seqNr,
      persistenceId = origPr.persistenceId,
      deleted = origPr.deleted,
      redeliveries = origPr.redeliveries,
      confirms = origPr.confirms,
      confirmable = origPr.confirmable)
    val msg = persistToJsonBytes(persistRepr) match {
      case Success(bytes) =>
        updateDb(origPr.persistenceId, seqNr, bytes)
        s"""json to be serialized to postgres 
                 |${Json.parse(bytes)}
                 |""".stripMargin
        
      case Failure(ex) =>
        log.error("Failed to restore $objType due to $ex", ex)
    }
    log.info(
      s"""
         |${`====`}
         |${getClass.getSimpleName}: EntryRetrieved received, id = $id, $objType = $obj
         |$msg
         |${`====`}
         |""".stripMargin)
  }
 
  val serialization = SerializationExtension(context.system)
 
  def persistToJsonBytes(js: AnyRef): Try[Array[Byte]] = serialization.serialize(js)
 
  def restoredFromJsonBytes(bytes: Array[Byte]): AnyRef = serialization.deserialize(bytes, classOf[PersistentRepr]).get
 
  val receive: Receive = {
 
    case NoEntriesFound =>
      toDelete.map(DeleteEntry).foreach(context.parent ! _)
      log.info(s"Done, noOfEntries = $noOfEntries")
      if (insertValuesForBatch.size > 0)
        executePstmt()
      Await result (future flatMap (_ => conn.disconnect), 10.minutes)
      log.info("all written to postgres")
 
    case EntryRetrieved(_, id, pr: PersistentRepr) =>
      noOfEntries = noOfEntries + 1
      
      val serializePersistentRepr = processPersistentRepr(id)(pr) _
      pr.payload match {
        case ARProtocol.PersistWrapper(arEvent, seqNr) =>
          arEventMatch(seqNr)(arEvent) map serializePersistentRepr("AREvent")("^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^^0^")
 
        case wrapper: DomainIndexProtocol.PersistWrapper =>
          serializePersistentRepr("DomainIndexProtocol.PersistWrapper")("=.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.==.=")(wrapper)
 
        case x => 
          log.error(
            s"""
               |----------------------------------
               |unprocessed data
               |$x
               |----------------------------------
             """.stripMargin)
      }
 
    case _ => ()
 
  }
 
}
```
In the above example, the actor inserts a record to database when it receives a message.  To avoid blocking the message reception, inserting a record to database is done in an asynchronous manner.  As this program inserts millions of records to the database, batch insertion is used for improving performance.  It will update the database only when it has received certain number of messages.  This is controlled by ```var insertValuesForBatch```, ```var paramsForBatch``` and ```val batchSize```.  ```executePstmt()``` does database update, clears ```insertValuesForBatch``` and ```paramsForBatch``` after update.  The update is done in an asychronous manner using a ```Future```.  It is very likely the variables are cleared before the ```Future``` is completed.  To avoid the problem, these variables should be assigned to ```val pStmt``` and ```val params```.

**Never use ```var``` directly inside a ```Future```**.

##Using => in different scenarios to achieve different purposes
**=>** is used in 3 scenarios which apparently look similar but actually is for different purpose.
Scenario 1
```Scala
trait Generator[+T] {
  blah => // an alias for ”this”.
  def generate: T
  def map[S](f: T => S): Generator[S] = new Generator[S] {
    def generate = f(blah.generate)
  }
}
```
```blah``` refers to the instance outside ```map```.  If ```f(this.generate)``` is used, it will point to the ```new Generate[S]instance```.
Scenario 2
```Scala
trait BoxOfficeCreator { this: Actor =>
  def createBoxOffice:ActorRef = {
     context.actorOf(Props[BoxOffice], "boxOffice")
  }
}
```
It means the trait uses self-type, that is, it should be mixed-in an ```Actor``` subclass.  Example
```Scala
class RestApi extends HttpService
  with ActorLogging
  with BoxOfficeCreator
  with Actor {
  // ...
}
```
Scenario 3
Regarding ```RestApi```, it can be implemented as
```Scala
class RestApi extends HttpService
  with ActorLogging
  with BoxOfficeCreator { blah: Actor =>
  // ...
}
```

