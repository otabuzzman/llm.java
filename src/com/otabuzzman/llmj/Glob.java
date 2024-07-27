package com.otabuzzman.llmj;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

public class Glob {

    private List<String> gl_pathv = new ArrayList<String>();

    public Glob(String pattern, String path) throws IOException {
        try { // 1st check path/pattern for existence
            String filename = FileSystems.getDefault().getPath(path, pattern).toString();
            if (new File(filename).exists()) {
                gl_pathv.add(filename);
            }
        } catch (InvalidPathException e) {
            PathMatcher matcher = FileSystems.getDefault().getPathMatcher("glob:" + pattern);

            Files.walkFileTree(Paths.get(path), new SimpleFileVisitor<Path>() {

                @Override
    			public FileVisitResult visitFile(Path path, BasicFileAttributes attributes) throws IOException {
    				if (matcher.matches(path)) {
                        gl_pathv.add(path.toString());
    				}
    				return FileVisitResult.CONTINUE;
    			}

    			@Override
    			public FileVisitResult visitFileFailed(Path path, IOException exception) throws IOException {
    				return FileVisitResult.CONTINUE;
    			}
            });
        }
    }

    public String gl_pathv(int index) {
        return gl_pathv.get(index);
    }

    public int gl_pathc() {
        return gl_pathv.size();
    }
}
