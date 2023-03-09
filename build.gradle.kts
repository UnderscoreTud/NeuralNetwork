plugins {
    id("java")
    id("maven-publish")
}

group = "me.tud"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains:annotations:20.1.0")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.1")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.8.1")
}

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "me.tud"
            artifactId = "NeuralNetwork"
        }
    }
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}