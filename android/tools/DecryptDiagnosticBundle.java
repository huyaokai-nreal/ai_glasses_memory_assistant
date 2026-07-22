import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.Console;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import javax.crypto.Cipher;
import javax.crypto.CipherInputStream;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.PBEKeySpec;
import javax.crypto.spec.SecretKeySpec;

public final class DecryptDiagnosticBundle {
    private static final byte[] MAGIC = "AIGDIAG1".getBytes(StandardCharsets.US_ASCII);
    private static final int MAX_ITERATIONS = 2_000_000;

    private DecryptDiagnosticBundle() {}

    public static void decrypt(Path source, Path destination, char[] password) throws Exception {
        Path temporary = null;
        try {
            if (!Files.isRegularFile(source)) {
                throw new IllegalArgumentException("diagnostic bundle does not exist: " + source);
            }
            if (Files.exists(destination)) {
                throw new IllegalArgumentException("refusing to overwrite existing output: " + destination);
            }
            if (password.length == 0) {
                throw new IllegalArgumentException("password is required");
            }
            Path parent = destination.toAbsolutePath().getParent();
            if (parent != null) Files.createDirectories(parent);
            temporary = Files.createTempFile(parent, ".diagnostic-", ".tmp");
            try (DataInputStream encrypted = new DataInputStream(
                new BufferedInputStream(new FileInputStream(source.toFile())))) {
            byte[] magic = encrypted.readNBytes(MAGIC.length);
            if (!Arrays.equals(magic, MAGIC)) {
                throw new IllegalArgumentException("not an AIGDIAG1 diagnostic bundle");
            }
            int iterations = encrypted.readInt();
            if (iterations <= 0 || iterations > MAX_ITERATIONS) {
                throw new IllegalArgumentException("invalid PBKDF2 iteration count");
            }
            byte[] salt = readSized(encrypted, 16, "salt");
            byte[] iv = readSized(encrypted, 12, "IV");
            PBEKeySpec specification = new PBEKeySpec(password, salt, iterations, 256);
            byte[] keyBytes;
            try {
                keyBytes = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
                        .generateSecret(specification)
                        .getEncoded();
            } finally {
                specification.clearPassword();
            }
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(keyBytes, "AES"), new GCMParameterSpec(128, iv));
            Arrays.fill(keyBytes, (byte) 0);
            try (CipherInputStream plaintext = new CipherInputStream(encrypted, cipher);
                 BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(temporary.toFile()))) {
                plaintext.transferTo(output);
            }
                try {
                    Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE);
                } catch (java.nio.file.AtomicMoveNotSupportedException ignored) {
                    Files.move(temporary, destination);
                }
            }
        } finally {
            Arrays.fill(password, '\0');
            if (temporary != null) Files.deleteIfExists(temporary);
        }
    }

    private static byte[] readSized(DataInputStream stream, int expected, String label) throws Exception {
        int size = stream.readUnsignedByte();
        if (size != expected) throw new IllegalArgumentException("invalid diagnostic " + label + " size");
        byte[] value = stream.readNBytes(size);
        if (value.length != size) throw new IllegalArgumentException("truncated diagnostic bundle header");
        return value;
    }

    private static char[] readPassword() throws Exception {
        Console console = System.console();
        if (console != null) return console.readPassword("Diagnostic password: ");
        System.err.print("Diagnostic password: ");
        return new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)).readLine().toCharArray();
    }

    public static void main(String[] args) {
        if (args.length != 1 && !(args.length == 3 && "--output".equals(args[1]))) {
            System.err.println("Usage: java DecryptDiagnosticBundle.java report.aigd [--output report.zip]");
            System.exit(2);
        }
        Path source = Path.of(args[0]).toAbsolutePath().normalize();
        Path destination = args.length == 3
                ? Path.of(args[2]).toAbsolutePath().normalize()
                : source.resolveSibling(source.getFileName().toString().replaceFirst("\\.aigd$", "") + ".zip");
        try {
            decrypt(source, destination, readPassword());
            System.out.println(destination);
        } catch (Exception error) {
            System.err.println("decrypt failed: " + error.getClass().getSimpleName() + ": " + error.getMessage());
            System.exit(1);
        }
    }
}
