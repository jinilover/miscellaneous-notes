# Note on Spark
Some simple but important note that helps me to have better understanding and beware of.

## What is Spark?
Spark is a clustering computing platform built on top of Akka.

## Its relationship with other jargons

### HDFS
HDFS is a distributed file system across different machines.  Using this distributed FS,  a huge file can be divided into chunks stored in different machines to enable parallel computing.  Each chunk can be replicated across different machines for failure recovery.  Distributed computation is executed on top of HDFS.  Spark and Hadoop are examples of these distributed computing platform.  It does not mean that Spark can run on top of HDFS only.  Besides HDFS, there are available distributed FS such as Google FS and CloudStore.

### Hadoop
It is a software platform for distributed storage and distributed processing.  It contains several components such as HDFS (mentioned in the previous section), YARN resource manager, Hadoop MapReduce.  Several of these components such as HDFS and YARN can work with Spark.  The Hadoop MapReduce is an implementation of the MapReduce programming model, i.e. a distributed processing which therefore can be regarded as Spark's competitor.

### MapReduce
It is a parallel processing algorithm.  Map and Reduce are inspired by the corresponding concept in functional programming.  The distributed computing platform, such as Spark, provide this algorithm implementation.  The Spark application developer only needs to provide the **map functions** (Spark transformation) and **reduce functions** (Spark action)for the platform which handles the rest of the processing.

## Why Spark?
Unlike Hadoop processing, after it reads data from disk and processes, it doesn't write data back to disk but store on memory instead.  This makes it perform faster.

## Basic structure
A spark application has a driver program that start up parallel operation on a cluster of compute machines.  This driver program can be regarded as the entry point.
Spark installation provides a spark shell which is a driver program.  Alternatively, a spark application can be written in a programming language such as Scala.  This programming application is a driver program.

## Spark Context
A connection to the Spark platform.  It is the entry point to build RDDs.

## RDD
Resilient Distributed Data.  It the core programming abstraction.  As it implies, it represents a collection of data distributed across different partition with each partition might be allocated to a machine such that the data items can be manipulated in parallel.

### Transform
It is transformation of RDD data items into another collection of data items by creating a new RDD.  It is like a ```map``` task in MapReduce.  Transformation is lazy evaluated.  Only action operation can trigger the transformation execution.  **Unlike the behaviour of Scala ```lazy val```, the transformation and its source transformation will always be evaluated when it is triggered by an action.  To improve the performance, the RDD's ```cache``` is provided to cache the transformation result.**
Example of API: ```map```, ```filter```, etc.

### Action
It is the operation that returns a value to the driver program by processing the RDD data of the machines.  It is like a ```reduce``` task in MapReduce.
Example of API: ```count```, ```collect```, ```reduce```, etc.
* ```collect``` returns all the data items from the whole RDD to the driver program.  Therefore this API should be used carefully when the RDD is huge
* ```reduce``` is applied at the partition level.  The aggregated value from each partition will be aggregated again to the driver program.  Therefore the function applied to ```reduce``` should be commutative and associative.

## Pair RDD
Operations on key/value pairs.  Partitioning is available on all Pair RDDs.
Example of transformation API: ```reduceByKey```, ```groupByKey```.
```reduceByKey``` and ```groupByKey``` are often used to solve the same problem and produce the same answer.  ```reduceByKey()``` transformation works better for large distributed datasets.  This is because Spark combines output with a common key on each partition before shuffling (redistributing) the data across machines.
That is, in ```reduceByKey```, pairs on the same machine with the same key are combined before the data is shuffled.
But in groupByKey(), all the key-value pairs are shuffled around, causing a lot of unnecessary data to being transferred over the network.

### Passing functions to partition
When a RDD higher-order function is called with a function literal, Sparks passes this function to all partition underlying.  That is, this function will be sent to different machines.  *This feature is not specific to Spark but also in other big data processing such as map reduce where computation is moved to the data node.*  If the function is a closure, the free variable should be passed.  Then a large overhead should be avoided in serializing this free variable which may involves hidden dependency.  There is a Scala macro called ```Spore``` that restricts the scope of the dependency graph in serializing the free variable for the closure.  [A talk by Martin Odersky, Spark -- the Ultimate Scala Collections](https://www.youtube.com/watch?v=NW5h8d_ZyOs).

### Some feature for improving performance
* Broadcast variables - send a value to each machine only once, it's a read-only variable.
* Accumulator - a write-only variable, this is to aggregate values from each machine and send back to the driver program.

## FP applicability
In processing BigData, all the operations are transforming and aggregating value.  Therefore FP is a good match for analyzing big data set.  
[A sample Spark application in Scala](https://github.com/jinilover/sparkPrototype).

## Spark SQL
Spark has several modules which base on top of Spark core.  The section discussed before are features of Spark core.  Spark SQL is one of the modules.  Spark SQL has nothing related to relational databse.  It is used to work with structured and semi-structured data.  A structured data format has a set of known fields for each record.  That's why it borrows the name 'SQL'.  E.g. tweets stored as Json format in a data storage.  Therefore using Spark SQL, it allows to write SQL-like statement such as 'select createdAt, text, user.location from *user-defined-table-name*'.  Therefore structured data are loaded and queried more efficiently by using Spark SQL.  E.g. the use of ```DataFrame``` which is a distributed collection of data organized into named columns.

## Spark Streaming
```DStream```, also called discretized stream, is the abstraction that a developer always works with.  It's called discretized stream becaues it represents a continuous stream of RDDs underlying.  DStream can come from any source such as socket, file, Twitter, etc.  

### Streaming Context
Analogous to Spark Context.  It is the entry point to build the stream.  E.g. ```aStreamContext.socketTextStream(host, port)``` builds a stream from a socket.
A stream context is created by ```new Streamingcontext(aSparkConf, Seconds(1))```.  **```Seconds(1)```** is the time interval such that those data available in this time interval is contained by the RDD for this time interval under the discretized stream.
* Those transformation/action operations applicable for RDD are also applicable for the stream.  However, these operations are applied **separately for each RDD but not across the RDDs under the stream**.
* The Stream context is actually created by a Spark context.
* A Spark context can be re-used to create multiple streaming context.  Since only 1 streaming context can be active in a JVM at the same time, a new streaming context can be created only after the other is stopped.
* stop() on a streaming context also stops the spark context behind.  Set the optional parameter of stop() if the spark context is re-used to create another streaming context.
* When a context has been stopped, it cannot be restarted.
* All streaming computation should be **set up** before a **streaming context** is started.  These setup for receiving this stream is "lazy" and reception starts only when the streaming context is started.
* A streaming context can created multiple streams.

### DStream
**Except file stream**, each DStream is associated with a Receiver which receives the data from the source and **stores it in Spark's memory for processing**.  http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.streaming.StreamingContext shows that ```fileStream``` returns ```InputDStream``` and the other sources returns ```ReceiverInputDStream```.  Those stream associates with a receiver should be allocated 2 cores or threads.  1 for receiving and storing the data in memory and the other for processing.

### Transformation
Given the nature of DStream, there are 2 kinds of transformations:
* Stateless transformation.  It refers to those transformations applied **separately for each RDD under the stream**.
* Stateful transformation.  It uses the data or intermediate result from previous RDDs to **compute the result of the current RDD**.

Types of **stateful transformations**:
* Windowed transformation - applied across the whole window that span across the RDDs inside the window.
* UpdateStateByKey transformation - applied across the whole stream.  This function requires to configure a checkpoint directory.

Similar to RDD transformation, DStream transformation is also lazy and executed only by DStream output operations (such as saving data) or having RDD action inside ```dstream.foreachRDD()```.  Otherwise it will receive the data only without any processing.

## References
* https://spark.apache.org/docs/latest/api/scala/#package
* https://spark.apache.org/docs/latest/streaming-programming-guide.html