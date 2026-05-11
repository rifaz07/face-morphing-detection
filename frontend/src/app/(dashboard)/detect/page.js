"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, ScanFace, X, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const MAX_SIZE_MB = 10;

export default function DetectPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null); // null | 'REAL' | 'MORPHED'

  const onDrop = useCallback((acceptedFiles) => {
    const f = acceptedFiles[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setPreview(URL.createObjectURL(f));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxSize: MAX_SIZE_MB * 1024 * 1024,
    multiple: false,
  });

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
  }

  // Placeholder — real FastAPI call wired in next feature
  function handleAnalyze() {
    setResult(Math.random() > 0.5 ? "REAL" : "MORPHED");
  }

  return (
    <div className="p-6 md:p-8 max-w-3xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Detect Morphed Face</h1>
        <p className="text-muted-foreground mt-1">
          Upload a face image to analyze it with our K-Means clustering pipeline.
        </p>
      </div>

      {/* Drop zone */}
      <Card
        {...getRootProps()}
        className={cn(
          "border-2 border-dashed cursor-pointer transition-all",
          isDragActive
            ? "border-purple-500 bg-purple-500/5"
            : "border-border/60 hover:border-purple-400/50 hover:bg-muted/30"
        )}
      >
        <CardContent className="py-14 flex flex-col items-center gap-4 text-center">
          <input {...getInputProps()} />
          <div className="rounded-full bg-purple-500/10 p-4">
            <Upload className="size-7 text-purple-500" aria-hidden />
          </div>
          {isDragActive ? (
            <p className="text-purple-500 font-medium">Drop the image here…</p>
          ) : (
            <>
              <p className="font-medium">Drag & drop a face image here</p>
              <p className="text-sm text-muted-foreground">
                or <span className="text-purple-500 underline">browse files</span>
              </p>
              <p className="text-xs text-muted-foreground">
                JPG, PNG, WEBP — max {MAX_SIZE_MB}MB
              </p>
            </>
          )}
        </CardContent>
      </Card>

      {/* Preview + result */}
      {preview && (
        <Card className="border-border/50 overflow-hidden">
          <CardContent className="p-5 space-y-5">
            <div className="flex items-start gap-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={preview}
                alt="Selected face"
                className="size-28 object-cover rounded-lg border border-border/50 shrink-0"
              />
              <div className="flex-1 min-w-0 space-y-1">
                <p className="font-medium truncate">{file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive/80 gap-1.5 -ml-2 mt-1"
                  onClick={clearFile}
                >
                  <X className="size-3.5" />
                  Remove
                </Button>
              </div>
            </div>

            <Button
              className="w-full gradient-brand text-white hover:opacity-90 gap-2"
              onClick={handleAnalyze}
            >
              <ScanFace className="size-4" />
              Analyze Image
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Detection result */}
      {result && (
        <Card
          className={cn(
            "border-2 transition-all",
            result === "REAL"
              ? "border-green-500/30 bg-green-500/5"
              : "border-red-500/30 bg-red-500/5"
          )}
        >
          <CardContent className="py-10 flex flex-col items-center gap-3 text-center">
            {result === "REAL" ? (
              <CheckCircle2 className="size-12 text-green-500" aria-hidden />
            ) : (
              <AlertCircle className="size-12 text-red-500" aria-hidden />
            )}
            <p className="text-2xl font-bold">
              {result === "REAL" ? "Real Face Detected" : "Morphed Face Detected"}
            </p>
            <p className="text-sm text-muted-foreground max-w-xs">
              {result === "REAL"
                ? "The image appears to be an authentic face image."
                : "This image shows signs of morphing attacks. Exercise caution."}
            </p>
            <p className="text-xs text-muted-foreground italic mt-1">
              Note: Full FastAPI integration coming in the next feature.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
