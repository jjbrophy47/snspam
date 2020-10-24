### PSL Installation ###

If maven is unable to retrieve the PSL 2.0.0 jar files for whatever reason, maven can manually install the necessary packages using the jar files in the `jars/` directory. Please run the following commands from within the `jars/` directory to install the PSL packages.

* `mvn install:install-file -DgroupId=org.linqs -DartifactId=psl-core -Dversion=2.0.0 -Dfile=psl-core-2.0.0.jar -DgeneratePom=true -Dpackaging=jar`
* `mvn install:install-file -DgroupId=org.linqs -DartifactId=psl-groovy -Dversion=2.0.0 -Dfile=psl-groovy-2.0.0.jar -DgeneratePom=true -Dpackaging=jar`
* `mvn install:install-file -DgroupId=org.linqs -DartifactId=psl-parser -Dversion=2.0.0 -Dfile=psl-parser-2.0.0.jar -DgeneratePom=true -Dpackaging=jar`
* `mvn install:install-file -DgroupId=org.linqs -DartifactId=psl-evaluation -Dversion=1.0.0 -Dfile=psl-evaluation-1.0.0.jar -DgeneratePom=true -Dpackaging=jar`
* `mvn install:install-file -DgroupId=org.linqs -DartifactId=psl-dataloading -Dversion=1.0.0 -Dfile=psl-dataloading-1.0.0.jar -DgeneratePom=true -Dpackaging=jar`

Required
--------
Java 1.8+ (i.e. /Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home/)
